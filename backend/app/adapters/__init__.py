"""Modality adapters: turn a raw input into a normalised EvidenceBundle.

``image_adapter`` (image+caption) and ``video_adapter`` (frames + audio) land in Weeks 1
and 3 respectively. Both emit the same :class:`app.models.EvidenceBundle` (ADR-003).
"""
