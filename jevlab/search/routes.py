"""Which generator models are asked to write which tactics.

Dear models stay in the pool, but a batch only goes to one when the tactics drawn
for that batch are tactics it has actually beaten the cheap pool at. Until a
bake-off fills ``DEAR_ARCHETYPES``, nothing is filtered.
"""

from __future__ import annotations

# OpenRouter rates put these well above the flash models. The route bake-off
# (scripts/route_bakeoff.py) decides which tactics are worth that price.
DEAR_MODELS = frozenset({"anthropic/claude-sonnet-5.5"})

# None: not tuned, every model writes whatever the batch drew.
# A set: the only tactics a dear model is asked for. Empty means it is never called.
DEAR_ARCHETYPES: frozenset[str] | None = None


def assign(model: str, archetypes: list[str]) -> list[str] | None:
    """Tactics ``model`` should write for this batch. None means skip the model."""
    if DEAR_ARCHETYPES is None or model not in DEAR_MODELS:
        return list(archetypes)
    kept = [name for name in archetypes if name in DEAR_ARCHETYPES]
    return kept or None
