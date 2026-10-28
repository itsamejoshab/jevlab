"""Usage-sense glyphs from a cached BERTweet masked-word pass.

The probe ranks catalog name-words that the winning line lifts above the
bland sentence "the answer is fine". It returns atoms. It does not score them.
"""

from __future__ import annotations

from .lexicon import Index
from .rank import Atom

_PROBLEM: str | None = None
_TRIED = False
_BUNDLE = None


def usage_problem() -> str | None:
    """'model not cached' after a failed load, else None."""
    return _PROBLEM


def _load():
    """Tokenizer and masked LM from the local Hugging Face cache. None when it is not there."""
    global _PROBLEM, _TRIED, _BUNDLE
    if _TRIED:
        return _BUNDLE
    _TRIED = True
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception:
        _PROBLEM = "model not cached"
        _BUNDLE = None
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base", local_files_only=True)
        model.eval()
    except Exception:
        _PROBLEM = "model not cached"
        _BUNDLE = None
        return None
    _PROBLEM = None
    _BUNDLE = (tokenizer, model)
    return _BUNDLE


def _mask_logprobs(tokenizer, model, text: str):
    import torch

    masked = f"{text.rstrip()} {tokenizer.mask_token}"
    encoded = tokenizer(masked, return_tensors="pt")
    mask_id = tokenizer.mask_token_id
    positions = (encoded["input_ids"][0] == mask_id).nonzero(as_tuple=False)
    if len(positions) == 0:
        return None
    index = int(positions[0])
    with torch.no_grad():
        logits = model(**encoded).logits[0, index]
        return torch.log_softmax(logits, dim=-1)


def usage_glyphs(phrase: str, index: Index, k: int = 8) -> list[Atom]:
    """Atoms whose name word is a single BERTweet piece and lifts above the bland sentence."""
    global _PROBLEM
    bundle = _load()
    if bundle is None:
        return []
    tokenizer, model = bundle
    try:
        phrase_logp = _mask_logprobs(tokenizer, model, phrase)
        base_logp = _mask_logprobs(tokenizer, model, "the answer is fine")
    except Exception:
        _PROBLEM = "model not cached"
        return []
    if phrase_logp is None or base_logp is None:
        return []
    _PROBLEM = None
    lifts: list[tuple[float, str]] = []
    for token in index.by_token:
        piece_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(piece_ids) != 1:
            continue
        piece = piece_ids[0]
        lift = float(phrase_logp[piece] - base_logp[piece])
        if lift > 0:
            lifts.append((lift, token))
    lifts.sort(key=lambda item: item[0], reverse=True)
    chosen: list[Atom] = []
    seen: set[str] = set()
    for _lift, token in lifts:
        for atom in index.by_token[token]:
            if atom.text in seen:
                continue
            seen.add(atom.text)
            chosen.append(atom)
            break
        if len(chosen) >= k:
            break
    return chosen
