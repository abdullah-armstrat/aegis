"""The fusion core: turns an Evidence Bundle into flags.

``rules`` holds the deterministic checks, ``llm_reasoner`` asks the local LLM about the supplied
text, and ``scorecard`` puts the flags together. The rules are the baseline and the LLM only
adds to them.
"""
