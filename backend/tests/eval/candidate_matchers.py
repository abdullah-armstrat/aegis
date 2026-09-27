"""Second-stage image matchers compared in WP-1b, for when the perceptual hash finds nothing.

pHash is excellent against recompression and resizing but loses an image once it is framed:
a border or a phone-screenshot frame changes the low frequencies the hash is built from. Three
candidates for a second stage are compared here, each run only on a lookup the hash missed:

  trim   Cut away flat margins (uniform borders, and the mostly-flat bars of a screenshot frame)
         until the photo region is left, then hash that region with the same pHash.
  orb    ORB keypoints matched to each known image, confirmed by a RANSAC homography: a match
         needs enough keypoint pairs that agree on one geometric mapping between the two images.
  clip   Cosine similarity of CLIP ViT-B/32 image embeddings.

ORB + RANSAC won the comparison and now lives in ``app/extractors/keypoint_match.py``; it is
imported from there below. trim and clip stay here as the record of what else was measured.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# --- trim --------------------------------------------------------------------------------------
FLAT_SHARE = 0.55   # a row or column is "flat" when at least this share of it is one colour
FLAT_TOLERANCE = 12  # grey levels within which pixels count as that colour
MIN_KEEP = 0.25      # never trim below a quarter of the original width or height


def _flat(line: np.ndarray) -> bool:
    values, counts = np.unique(line // FLAT_TOLERANCE, return_counts=True)
    return counts.max() / line.size >= FLAT_SHARE


def trim_to_content(img: Image.Image) -> Image.Image:
    """Remove flat rows and columns from every edge, leaving the photo region."""
    grey = np.asarray(img.convert("L"), dtype=np.int16)
    h, w = grey.shape
    top, bottom, left, right = 0, h, 0, w
    for _ in range(2):  # columns, then rows, twice: frames are trimmed from the outside in
        while right - left > w * MIN_KEEP and _flat(grey[top:bottom, left]):
            left += 1
        while right - left > w * MIN_KEEP and _flat(grey[top:bottom, right - 1]):
            right -= 1
        while bottom - top > h * MIN_KEEP and _flat(grey[top, left:right]):
            top += 1
        while bottom - top > h * MIN_KEEP and _flat(grey[bottom - 1, left:right]):
            bottom -= 1
    return img.convert("RGB").crop((left, top, right, bottom))


# --- orb ---------------------------------------------------------------------------------------
# The chosen matcher lives in the application; importing it here means the comparison measures
# exactly the code that ships.
from app.extractors.keypoint_match import orb_features, orb_inliers  # noqa: E402,F401


# --- clip --------------------------------------------------------------------------------------
class ClipEmbedder:
    """CLIP ViT-B/32 image embeddings, unit length, on CPU."""

    def __init__(self, arch: str = "ViT-B/32"):
        import clip
        import torch

        self._torch = torch
        self.model, self.preprocess = clip.load(arch, device="cpu")
        self.model.eval()

    def embed(self, img: Image.Image) -> np.ndarray:
        with self._torch.no_grad():
            vec = self.model.encode_image(self.preprocess(img.convert("RGB")).unsqueeze(0))[0]
        vec = vec.float().numpy()
        return vec / np.linalg.norm(vec)
