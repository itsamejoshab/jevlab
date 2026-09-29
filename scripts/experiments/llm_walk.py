"""End-append walk where an LLM proposes coherent next words (the leader's lines read as text, not salad).
Keeps any word that holds the rounded score, ranking holders by a probe, and backs off when nothing holds."""
import asyncio, collections, json, random, re, sys
from jevlab.db import DB
from jevlab.llm import LLM
from jevlab.oracle import Oracle, question_key
from jevlab.rules import normalize

BOARDS = {"do-you-love-me": 62, "will-you-accept-my-job-application": 68, "is-git-gud-useful-advice": 63}
BOARDS = {k: v for k, v in BOARDS.items() if k in sys.argv[1].split(",")}
BUDGET = int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
MODEL = "google/gemini-3.8-flash"


def prompt(title, answer, text, n):
    return (f"A judge will read a statement and answer the question \"{title}\" with {answer.upper()} or the "
            f"opposite. The statement so far is:\n\n{text}\n\nPropose {n} different single words that could come next, "
            f"each continuing the statement naturally as English text so that it pushes the judge ever more "
            f"certainly to {answer.upper()}. Mix styles: continuing the sentence, starting a new emphatic clause, "
            f"stating a rule or fact that settles it. One word each, no punctuation. "
            f"JSON: {{\"words\": [..]}}")


async def walk(db, llm, slug, lead):
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (slug,))[0])
    req = json.loads(q["jev_request"])
    goal = q["goal"] or "yes"
    gp = (lambda v: v) if goal != "no" else (lambda v: 1 - v)
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    tops = [s for s, v in per.items() if min(v) >= 0.98 and len(s.split()) <= 20 and len(v) >= 2]
    root = rng.choice(sorted(tops, key=lambda s: -len(per[s]))[:40])
    o = Oracle(db, req)
    opp = "yes" if goal == "no" else "no"
    base = f"{opp} {normalize(q['title']).rstrip('?')} {opp}"
    claim = None
    for k in (1, 2, 4, 8):
        c = " ".join([base] * k)
        s = await o.score([f"{root} {c}"], 2, qkey=f"{o.qkey}:probe")
        v = gp(s[f"{root} {c}"].mean)
        if v <= 0.85:
            claim = c if v >= 0.35 else None
            break

    async def probe(ps):
        if not claim:
            return {p: rng.random() for p in ps}
        s = await o.score([f"{p} {claim}" for p in ps], 1, qkey=f"{o.qkey}:probe")
        return {p: gp(s[f"{p} {claim}"].mean) for p in ps}

    async def propose(line):
        try:
            data = await llm.json(MODEL, "Reply with JSON only.", prompt(q["title"], goal, " ".join(line), 20),
                                  temperature=1.0, max_tokens=1000)
        except Exception as e:
            print("  llm fail", repr(e)[:100]); return []
        out = []
        for w in (data or {}).get("words") or []:
            w = re.sub(r"[^A-Za-z0-9']", "", str(w)).lower()
            if w and w not in out:
                out.append(w)
        return out[:16]

    line, level, misses, hits, longest, ups = root.split(), 0.98, 0, [], len(root.split()), 0
    print(f"## {slug} root {len(line)}w, probe {'on' if claim else 'off'}: {root}", flush=True)
    while o.calls < BUDGET and len(line) < lead - 1:
        words = await propose(line)
        if not words:
            continue
        cands = [" ".join(line + [w]) for w in words]
        s = await o.score(cands, 1)
        vals = {p: round(gp(s[p].mean), 2) for p in cands if s[p].n}
        up = [p for p, v in vals.items() if v > level]
        if up:
            ups += len(up)
            again = await o.score(up, 4)
            for p in up:
                rolls = [round(gp(x), 2) for x in again[p].samples]
                print(f"   UP {len(p.split())}w rolls {rolls} :: {p}", flush=True)
                if sum(r >= 0.99 for r in rolls) >= 3:
                    hits.append(p)
            if hits:
                break
        hold = [p for p, v in vals.items() if v >= level]
        if hold:
            pv = await probe(hold)
            best = max(hold, key=lambda p: pv[p])
            line, misses = best.split(), 0
            longest = max(longest, len(line))
        else:
            misses += 1
            if misses >= 2 and len(line) > len(root.split()):
                line, misses = line[:-1], 0
    print(f"   end: {o.calls} calls, reached {longest}w, {ups} single 0.99 reads, {len(hits)} confirmed under {lead}w",
          flush=True)
    print(f"   final: {' '.join(line)}", flush=True)
    await o.close()


async def main():
    db, llm = DB(), LLM()
    for slug, lead in BOARDS.items():
        await walk(db, llm, slug, lead)


asyncio.run(main())
