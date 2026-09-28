"""Keypoint matching: the second stage of the recycled-context lookup.

The perceptual hash misses copies that are cropped, bordered or framed in a screenshot. Here ORB
keypoints are paired between the two images and a RANSAC homography keeps the pairs that agree on
one mapping, so many agreeing pairs mean the same picture. It is slower, so it only runs when the
hash finds nothing. Very low-texture images (heavy blur, flat graphics) give too few keypoints.
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
    """Return (keypoint coordinates, binary descriptors) for the image scaled to ORB_MAX_SIDE."""
    import cv2

    grey = np.asarray(img.convert("L"))
    scale = ORB_MAX_SIDE / max(grey.shape)
    if scale < 1:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, desc = cv2.ORB_create(nfeatures=ORB_FEATURES).detectAndCompute(grey, None)
    return np.float32([k.pt for k in keypoints]), desc


def orb_inliers(query: tuple, entry: tuple) -> int:
    """Number of one-to-one keypoint pairs that agree on a single homography."""
    return orb_homography(query, entry)[0]


def orb_homography(query: tuple, entry: tuple) -> tuple[int, np.ndarray | None]:
    """Return (agreeing one-to-one pairs, homography from query to entry, or None).

    The homography works on the images at ORB's working size (see ``working_size``).
    """
    import cv2

    (q_pts, q_desc), (e_pts, e_desc) = query, entry
    if q_desc is None or e_desc is None or len(q_desc) < MIN_PAIRS or len(e_desc) < MIN_PAIRS:
        return 0, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(q_desc, e_desc, k=2)
    # Lowe's ratio test drops ambiguous matches
    good = [m for m, *rest in pairs if rest and m.distance < RATIO_TEST * rest[0].distance]
    # Keep matches one-to-one. Otherwise many query points pile onto the few keypoints of a
    # low-texture image: one unrelated pair scored 62 inliers this way, and 6 with this step.
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
    """The image as an RGB array at the size ``orb_features`` works on."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    scale = ORB_MAX_SIDE / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return rgb


def aligned_distance(query: Image.Image, entry: Image.Image, homography: np.ndarray) -> int:
    """pHash distance between the entry and the query warped onto it, over the area the query covers.

    Keypoints can also match on something two different photos share (the same building on another
    day), so a keypoint match must pass the hash threshold here too. Both images are cropped to the
    covered box, and pixels outside the covered area are set to the same grey.
    """
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
