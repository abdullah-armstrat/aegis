"""Keypoint matching — the second stage of the recycled-context lookup.

The perceptual hash finds recompressed and resized copies, but loses an image once it is framed,
bordered or cropped: those change the low frequencies the hash is built from. Keypoint matching
does not depend on them. ORB finds distinctive corners in both images, pairs them by descriptor,
and a RANSAC homography then keeps only the pairs that agree on one geometric mapping between the
two images. Many agreeing pairs mean the same picture, even inside a screenshot frame.

It runs only when the hash lookup finds nothing, because it is slower and scales with the number
of known images.

Two details keep unrelated images apart:
  * Lowe's ratio test drops ambiguous descriptor matches.
  * Matches are one-to-one: each known-image keypoint may be claimed by one query keypoint only.
    Without this, many query points pile onto the few keypoints of a low-texture image and
    RANSAC counts them all; in the comparison that produced this module, that raised the score
    of an unrelated pair to 62, where one-to-one matching gives 6.

Images with very little texture (a heavily blurred photo, a flat graphic) yield too few keypoints
to match; the hash stage is the only one that can find them.

Keypoints can also agree on something two different photos share, such as the same building photographed
on different days. So a keypoint match is confirmed by alignment (ADR-056): the copy is warped onto
the candidate with the homography the points give, and the region it covers must pass the same
hash threshold as the first stage. A different photo of the same building differs there (the sky,
the light, the foreground); a copy does not.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ORB_FEATURES = 1000
ORB_MAX_SIDE = 640   # longer image side is scaled down to this before detection
RATIO_TEST = 0.75    # Lowe's ratio on the two nearest descriptor matches
RANSAC_PIXELS = 5.0  # reprojection error allowed for a pair to count as agreeing
MIN_PAIRS = 8        # fewer candidate pairs than this cannot support a homography


def orb_features(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """(keypoint coordinates, binary descriptors) for an image, at a bounded size."""
    import cv2

    grey = np.asarray(img.convert("L"))
    scale = ORB_MAX_SIDE / max(grey.shape)
    if scale < 1:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, desc = cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(grey, None)
    return np.float32([k.pt for k in keypoints]), desc


def orb_inliers(query: tuple, entry: tuple) -> int:
    """How many one-to-one keypoint pairs agree on a single homography between two images."""
    return orb_homography(query, entry)[0]


def orb_homography(query: tuple, entry: tuple) -> tuple[int, np.ndarray | None]:
    """(agreeing one-to-one pairs, the homography taking the query onto the entry at ORB's working
    size, or None when there is none)."""
    import cv2

    (q_pts, q_desc), (e_pts, e_desc) = query, entry
    if q_desc is None or e_desc is None or len(q_desc) < MIN_PAIRS or len(e_desc) < MIN_PAIRS:
        return 0, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(q_desc, e_desc, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < RATIO_TEST * rest[0].distance]
    best: dict[int, object] = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if len(good) < MIN_PAIRS:
        return 0, None
    src = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = e_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PIXELS)
    return (int(mask.sum()), homography) if mask is not None else (0, None)


def working_size(img: Image.Image) -> np.ndarray:
    """The image as RGB at the size ``orb_features`` finds keypoints on."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    scale = ORB_MAX_SIDE / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return rgb


def aligned_distance(query: Image.Image, entry: Image.Image, homography: np.ndarray) -> int:
    """pHash distance between the candidate and the copy aligned onto it, over the region the copy
    covers: both cropped to that region's bounding box, with the pixels outside it the same grey."""
    import cv2

    from app.extractors.phash import HASH_BITS, hamming, phash_of_image

    q, e = working_size(query), working_size(entry)
    h, w = e.shape[:2]
    warped = cv2.warpPerspective(q, homography, (w, h))
    covered = cv2.warpPerspective(np.full(q.shape[:2], 255, np.uint8), homography, (w, h)) > 127
    if not covered.any():
        return HASH_BITS
    ys, xs = np.nonzero(covered)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    a, b, inside = e[box].copy(), warped[box].copy(), covered[box]
    a[~inside] = 128
    b[~inside] = 128
    return hamming(phash_of_image(Image.fromarray(a)), phash_of_image(Image.fromarray(b)))
