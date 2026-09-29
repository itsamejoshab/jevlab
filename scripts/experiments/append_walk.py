"""The leader's apparent method: a deep walk of end-appends from a strong line, keeping any word that holds the
rounded score (ties broken by a probe), backing off a word when nothing holds, up to the leader's length."""
import asyncio, collections, json, random, re, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key
from jevlab.rules import normalize

BOARDS = {"do-you-love-me": 62, "will-you-accept-my-job-application": 68, "is-git-gud-useful-advice": 63}
if len(sys.argv) > 1:
    BOARDS = {k: v for k, v in BOARDS.items() if k in sys.argv[1].split(",")}
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
K = 14
LEADER = ("rule rules 1 2 3 4 5 6 7 every always only never not no exceptions logic puzzle true axiom premise "
          "means world text all each must").split()
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)


async def walk(db, slug, lead):
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (slug,))[0])
    req = json.loads(q["jev_request"])
    goal = q["goal"] or "yes"
    gp = (lambda v: v) if goal != "no" else (lambda v: 1 - v)
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    tops = [s for s, v in per.items() if min(v) >= 0.98 and len(s.split()) <= 20 and len(v) >= 2]
    root = rng.choice(sorted(tops, key=lambda s: -len(per[s]))[:40])
    impacts = json.loads((db.one("SELECT rows FROM word_impacts WHERE slug=? AND mode='strict_chain'", (slug,))
                          or {"rows": "[]"})["rows"])
    site = [r["word"] for r in sorted(impacts, key=lambda r: -(r.get("tries") or 0))
            if (r.get("averageImpact") or 0) > 0 and " " not in r["word"] and re.fullmatch(r"[A-Za-z0-9]+", r["word"])][:150]
    ours = [w for w, _ in collections.Counter(w for s in tops[:300] for w in s.split()).most_common(80)]
    title = re.findall(r"[A-Za-z0-9]+", q["title"])
    pool = list(dict.fromkeys(site + LEADER + title + ours))
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

    line, level, misses, hits, longest = root.split(), 0.98, 0, [], len(root.split())
    print(f"## {slug} root {len(line)}w, probe {'on' if claim else 'off'}, pool {len(pool)}: {root}", flush=True)
    while o.calls < BUDGET and len(line) < lead - 1:
        words = rng.sample(pool, K)
        cands = [" ".join(line + [w]) for w in words]
        s = await o.score(cands, 1)
        vals = {p: round(gp(s[p].mean), 2) for p in cands if s[p].n}
        up = [p for p, v in vals.items() if v > level]
        if up:
            again = await o.score(up, 4)
            for p in up:
                rolls = [round(gp(x), 2) for x in again[p].samples]
                print(f"   UP {len(p.split())}w rolls {rolls} :: {p}", flush=True)
                if min(rolls) >= 0.99:
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
    print(f"   end: {o.calls} calls, reached {longest}w, line now {len(line)}w, {len(hits)} confirmed 0.99 "
          f"under {lead}w", flush=True)
    for p in hits:
        print(f"   WIN {len(p.split())}w :: {p}", flush=True)
    await o.close()
    return hits


async def main():
    db = DB()
    for slug, lead in BOARDS.items():
        await walk(db, slug, lead)


asyncio.run(main())
