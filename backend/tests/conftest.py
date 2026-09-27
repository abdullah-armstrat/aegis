"""Shared test configuration.

Keeps the fast suite fast: the BLIP captioner is enabled by default in production
(``AEGIS_USE_CAPTIONER=true``), but loading it costs ~1GB and seconds, so during
tests it defaults to **off** unless a test sets the variable itself. Tests that specifically
exercise the captioner either set ``AEGIS_USE_CAPTIONER`` explicitly (the adapter tests) or
are marked ``slow`` (the captioner extractor test, the eval before/after).

The same holds for the caption-vs-picture check: comparing by meaning loads CLIP and spaCy, so
tests default to the word-overlap method unless they set ``AEGIS_CAPTION_MATCH_METHOD``
themselves. The pinned 19-example regression matrix is the word-overlap rule's.

This only affects the default; a test that opts in by setting the env var still wins, and the
production default in ``config.py`` is unchanged.
"""

import os

# Set before any test imports trigger a settings read. Only set if the caller hasn't.
os.environ.setdefault("AEGIS_USE_CAPTIONER", "false")
os.environ.setdefault("AEGIS_CAPTION_MATCH_METHOD", "overlap")
