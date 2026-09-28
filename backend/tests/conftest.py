"""Shared test settings.

BLIP, CLIP and spaCy are slow to load, so tests default to no captioner and the word-overlap
caption check. A test can still turn either on by setting the env var itself.
"""

import os

# Set before any test import reads the settings, and only if not already set.
os.environ.setdefault("AEGIS_USE_CAPTIONER", "false")
os.environ.setdefault("AEGIS_CAPTION_MATCH_METHOD", "overlap")

# No test should make a real Google Vision call, so the key is removed for the whole run.
# Tests of the live path set a fake key and replace the network.
os.environ.pop("GOOGLE_VISION_API_KEY", None)
