"""The site's per-word tallies for a board include every word the leader appended. Treat them as a bag: build long
attempts near the leader's length from it (random draws weighted by tries, and LLM text restricted to it), then
prune the best down a word at a time, keeping deletions that hold, watching for the next level."""
import asyncio, collections, json, random, re, sys
from jevlab.db import DB
from jevlab.llm import LLM
from jevlab.oracle import Oracle, question_key
from jevlab.rules import RuleError, check_phrase, normalize

SLUG, LEAD = sys.argv[1], int(sys.argv[2])
N_RANDOM = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_LLM_CALLS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
PRUNE_BUDGET = int(sys.argv[5]) if len(sys.argv) > 5 else 800
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 5)
MODELS = ["google/gemini-3.8-flash", "deepseek/deepseek-v4-flash"]


def bag_of(db, slug):
    tries = collections.Counter()
    for mode in ("strict_chain", "word_chain"):
        row = db.one("SELECT rows FROM word_impacts WHERE slug=? AND mode=?", (slug, mode))
        for r in json.loads(row["rows"]) if row else []:
            w = str(r["word"])
            if re.fullmatch(r"[A-Za-z0-9']+", w):
                tries[w.lower()] += r.get("tries") or 1
    return tries


async def main():
    db, llm = DB(), LLM()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    req = json.loads(q["jev_request"])
    goal = q["goal"] or "yes"
    gp = (lambda v: 1 - v) if goal == "no" else (lambda v: v)
    tries = bag_of(db, SLUG)
    words, weights = list(tries), list(tries.values())
    print(f"## {SLUG}: bag {len(words)} words, {sum(weights)} tries; top {tries.most_common(25)}", flush=True)
    o = Oracle(db, req)
    lengths = lambda: rng.randint(LEAD + 1, LEAD + 12)

    randoms = {" ".join(rng.choices(words, weights, k=lengths())) for _ in range(N_RANDOM)}
    bag_text = " ".join(w for w, _ in tries.most_common(250))
    user = (f"A judge answers the question \"{q['title']}\" with {goal.upper()} or the opposite after reading a "
            f"statement. Write 8 statements of {LEAD + 2} to {LEAD + 12} words that make {goal.upper()} certain. "
            f"Use almost only words from this list, repeating them freely (numbers allowed): {bag_text}. "
            f"Plain words, no punctuation. JSON: {{\"texts\": [..]}}")
    got = await asyncio.gather(*(llm.json(MODELS[i % 2], "Reply with JSON only.", user, temperature=1.0,
                                          max_tokens=6000) for i in range(N_LLM_CALLS)), return_exceptions=True)
    composed = set()
    for g in got:
        for t in (g.get("texts") or []) if isinstance(g, dict) else []:
            try:
                composed.add(check_phrase(normalize(str(t)), [], 400))
            except RuleError:
                pass
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    ours = sorted((s for s, v in per.items() if min(v) >= 0.98 and len(v) >= 2), key=lambda s: -len(per[s]))[:60]
    stacked = set()
    for _ in range(N_RANDOM // 2 if ours else 0):
        parts, n, want = [], 0, lengths()
        pieces = ours + list(composed)
        while n < want:
            piece = rng.choice(ours) if not parts or rng.random() < 0.6 else rng.choice(pieces)
            if piece in parts:
                continue
            parts.append(piece)
            n += len(piece.split())
        stacked.add(" ".join(" ".join(parts).split()[: want]))
    results = {}
    for label, pool in (("random", randoms), ("llm", composed), ("stacked", stacked)):
        s = await o.score(list(pool), 1)
        vals = {p: round(gp(s[p].mean), 2) for p in pool if s[p].n}
        results.update(vals)
        print(f"{label}: {len(vals)} lines, dist {sorted(collections.Counter(vals.values()).items(), reverse=True)[:6]}",
              flush=True)
    ranked = sorted(results, key=lambda p: -results[p])
    print(f"best {results[ranked[0]]} {len(ranked[0].split())}w :: {ranked[0]}", flush=True)

    start = o.calls
    wins = []
    for root in ranked[:8]:
        line, level = root.split(), results[root]
        while o.calls - start < PRUNE_BUDGET and len(line) > 5:
            idx = rng.sample(range(len(line)), min(16, len(line)))
            cands = list(dict.fromkeys(" ".join(line[:i] + line[i + 1:]) for i in idx))
            s = await o.score(cands, 1)
            vals = {p: round(gp(s[p].mean), 2) for p in cands if s[p].n}
            if not vals:
                break
            best = max(vals, key=lambda p: (vals[p], rng.random()))
            if vals[best] < level:
                break
            if vals[best] > level:
                print(f"   up {level} -> {vals[best]} at {len(best.split())}w :: {best}", flush=True)
                level = vals[best]
            line = best.split()
            if level >= 0.99 and len(line) < LEAD:
                again = await o.score([best], 5)
                rolls = [round(gp(x), 2) for x in again[best].samples]
                print(f"   CHECK {rolls} {len(line)}w", flush=True)
                if min(rolls) >= 0.99:
                    wins.append(best)
        print(f"   pruned to {len(line)}w at {level} :: {' '.join(line)}", flush=True)
    print(f"total calls {o.calls}, wins {len(wins)}")
    for w in wins:
        print("WIN", len(w.split()), "w ::", w)
    await o.close()


asyncio.run(main())
