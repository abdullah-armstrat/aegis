"""Modality adapters: turn a raw input into an EvidenceBundle.

``image_adapter`` handles an image and caption, ``video_adapter`` a video and caption. Both emit
the same :class:`app.models.EvidenceBundle`, so the fusion core does not care which input it got.
"""
