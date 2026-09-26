"""Shared test configuration.

Keeps the fast suite fast: the BLIP captioner is enabled by default in production
(``AEGIS_USE_CAPTIONER=true``), but loading it costs ~1GB and seconds, so during
tests it defaults to **off** unless a test sets the variable itself. Tests that specifically
exercise the captioner either set ``AEGIS_USE_CAPTIONER`` explicitly (the adapter tests) or
are marked ``slow`` (the captioner extractor test, the eval before/after).

This only affects the default; a test that opts in by setting the env var still wins, and the
production default in ``config.py`` is unchanged.
"""

import os

# Set before any test imports trigger a settings read. Only set if the caller hasn't.
os.environ.setdefault("AEGIS_USE_CAPTIONER", "false")
