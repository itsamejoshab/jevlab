"""Keep generating long lines (leader length to 1.5x) until one reads the next level, then prune it below the
leader's length while it holds. Each batch recombines chunks of the best long lines so far with our strong lines
and LLM text written from the board's site word bag."""
import asyncio, collections, json, random, re, sys
from jevlab.db import DB
from jevlab.llm import LLM
from jevlab.oracle import Oracle, question_key
from jevlab.rules import RuleError, check_phrase, normalize

SLUG, LEAD = sys.argv[1], int(sys.argv[2])
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
BATCH = 120
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 7)
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


def chunks(line, rng):
    words, out, i = line.split(), [], 0
    while i < len(words):
        k = rng.randint(4, 12)
        out.append(words[i:i + k])
        i += k
    return out


async def main():
    db, llm = DB(), LLM()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    req = json.loads(q["jev_request"])
    goal = q["goal"] or "yes"
    gp = (lambda v: 1 - v) if goal == "no" else (lambda v: v)
    o = Oracle(db, req)
    bag = " ".join(w for w, _ in bag_of(db, SLUG).most_common(250))
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    ours = sorted((s for s, v in per.items() if min(v) >= 0.98 and len(v) >= 2), key=lambda s: -len(per[s]))[:80]
    lo, hi = LEAD, int(LEAD * 1.5)
    seen: dict[str, float] = {}
    llm_texts: list[str] = []

    async def write(best):
        shown = "".join(f"\n- {p}" for p in best[:3])
        user = (f"A judge answers the question \"{q['title']}\" with {goal.upper()} or the opposite after reading a "
                f"statement. Write 8 statements of {lo} to {hi} words that make {goal.upper()} utterly certain. "
                f"Use mostly words from this list, repeating them freely: {bag}. Plain words, no punctuation."
                + (f"\nThese long ones scored best so far; write in their spirit but vary them:{shown}" if best else "")
                + " JSON: {\"texts\": [..]}")
        got = await asyncio.gather(*(llm.json(MODELS[i % 2], "Reply with JSON only.", user, temperature=1.0,
                                              max_tokens=8000) for i in range(2)), return_exceptions=True)
        for g in got:
            for t in (g.get("texts") or []) if isinstance(g, dict) else []:
                try:
                    llm_texts.append(check_phrase(normalize(str(t)), [], 400))
                except RuleError:
                    pass

    def make():
        want = rng.randint(lo, hi)
        elite = sorted((p for p in seen if len(p.split()) >= lo), key=lambda p: -seen[p])[:40]
        sources = [(ours, 0.4), (elite, 0.4), (llm_texts, 0.2)]
        out: list[str] = []
        while len(out) < want:
            pool = rng.choices([s for s, _ in sources if s], [w for s, w in sources if s])[0]
            line = rng.choice(pool)
            if rng.random() < 0.5:
                out += line.split()
            else:
                out += rng.choice(chunks(line, rng))
        return " ".join(out[:want])

    hit = None
    rnd = 0
    while o.calls < BUDGET and hit is None:
        rnd += 1
        if rnd % 3 == 1:
            await write(sorted((p for p in seen if len(p.split()) >= lo), key=lambda p: -seen[p]))
        cands = list({make() for _ in range(BATCH)} - set(seen))
        s = await o.score(cands, 1)
        for p in cands:
            if s[p].n:
                seen[p] = round(gp(s[p].mean), 2)
        dist = collections.Counter(seen[p] for p in cands if p in seen)
        print(f"round {rnd}: calls {o.calls}, batch top {max(dist)}, {dist.get(0.98, 0)} at 0.98, "
              f"{dist.get(0.99, 0)} at 0.99", flush=True)
        for p in [p for p in cands if seen.get(p, 0) >= 0.99]:
            again = await o.score([p], 5)
            rolls = [round(gp(x), 2) for x in again[p].samples]
            print(f"   0.99 read {rolls} {len(p.split())}w :: {p}", flush=True)
            if sum(r >= 0.99 for r in rolls) >= 4:
                hit = p
                break
    if hit is None:
        print("no 0.99 within budget")
        await o.close()
        return
    line = hit.split()
    print(f"pruning from {len(line)}w", flush=True)
    while len(line) > 5 and o.calls < BUDGET * 1.5:
        cands = list(dict.fromkeys(" ".join(line[:i] + line[i + 1:]) for i in range(len(line))))
        rng.shuffle(cands)
        moved = False
        for start in range(0, len(cands), 24):
            part = cands[start:start + 24]
            s = await o.score(part, 1)
            ok = [p for p in part if s[p].n and round(gp(s[p].mean), 2) >= 0.99]
            if ok:
                again = await o.score(ok[:3], 4)
                good = [p for p in ok[:3] if min(round(gp(x), 2) for x in again[p].samples) >= 0.99]
                if good:
                    line, moved = good[0].split(), True
                    print(f"   {len(line)}w :: {good[0]}", flush=True)
                    break
        if not moved:
            break
    print(f"final {len(line)}w (leader {LEAD}) :: {' '.join(line)}  calls {o.calls}")
    await o.close()


asyncio.run(main())
