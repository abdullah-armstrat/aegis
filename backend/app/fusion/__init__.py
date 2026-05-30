"""The fusion core — the project's intellectual heart.

``rules`` (deterministic, built first), then ``llm_reasoner`` (reasons over supplied text
only, ADR-005), then ``scorecard`` assembles flags. Rules ship before the LLM (ADR-004).
"""
