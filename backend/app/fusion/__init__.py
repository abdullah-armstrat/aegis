"""The fusion core — the project's intellectual heart.

``rules`` (deterministic, built first), then ``llm_reasoner`` (reasons over supplied text
only), then ``scorecard`` assembles flags. The rules are the reproducible baseline; the LLM
only adds to them.
"""
