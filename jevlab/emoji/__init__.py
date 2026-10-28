"""Emoji play mode: a local proposer and a capped oracle, separate from the Strict search."""

from .rules import RuleError, check_phrase, check_word
from .search import search

__all__ = ["RuleError", "check_phrase", "check_word", "search"]
