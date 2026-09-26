"""Single-modality extractors.

Each extractor does one reliable task and is added one at a time with its own unit test:
OCR -> sentiment -> captioner -> reverse_image (offline index) -> ai_gen_hint (weak).
"""
