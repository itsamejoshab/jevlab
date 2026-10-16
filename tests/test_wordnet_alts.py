"""WordNet adds a couple of synonyms behind the LLM alternative list."""

from jevlab.search.engine import WORDNET_ALTS, augment_alts, sprinkle_wordnet


def test_wordnet_appends_without_moving_the_llm_list():
    alts = ["spice", "flavor"]
    merged = augment_alts("season", alts)
    assert merged[:2] == ["spice", "flavor"]
    assert merged[2:] == ["flavour", "harden"]
    assert len(merged) - len(alts) == WORDNET_ALTS


def test_wordnet_is_applied_once_even_when_more_synonyms_exist():
    synonyms = {"season": ["spice", "flavor"]}
    done: set[str] = set()
    sprinkle_wordnet(synonyms, done, "season")
    once = list(synonyms["season"])
    sprinkle_wordnet(synonyms, done, "season")
    assert synonyms["season"] == once
    assert once == ["spice", "flavor", "flavour", "harden"]


def test_wordnet_skips_banned_words_and_keeps_going():
    merged = augment_alts("happy", ["joyful"], banned={"glad"})
    assert merged == ["joyful", "felicitous"]


def test_wordnet_leaves_the_list_alone_when_the_word_has_no_synonyms():
    assert augment_alts("zzzzqqq", ["kept"]) == ["kept"]
