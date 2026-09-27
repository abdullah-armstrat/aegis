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
    import cv2

    (q_pts, q_desc), (e_pts, e_desc) = query, entry
    if q_desc is None or e_desc is None or len(q_desc) < MIN_PAIRS or len(e_desc) < MIN_PAIRS:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(q_desc, e_desc, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < RATIO_TEST * rest[0].distance]
    best: dict[int, object] = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if len(good) < MIN_PAIRS:
        return 0
    src = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = e_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PIXELS)
    return int(mask.sum()) if mask is not None else 0
