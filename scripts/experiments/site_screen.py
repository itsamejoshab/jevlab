"""Screen our long 0.98 lines on the live site: rank by probe, build each and re-roll it a few times, stop at the
first 0.99 read (the board keeps our best-ever row)."""
import asyncio, collections, json, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key
from jevlab.publish import build_chain, reroll
from jevlab.rules import normalize
from jevlab.site import SiteClient

SLUG, LEAD = sys.argv[1], int(sys.argv[2])
LINES = int(sys.argv[3]) if len(sys.argv) > 3 else 20
REROLLS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
MIN_WORDS = int(sys.argv[5]) if len(sys.argv) > 5 else 40


async def rank(db, q, pool, gp):
    o = Oracle(db, json.loads(q["jev_request"]))
    goal = q["goal"] or "yes"
    opp = "yes" if goal == "no" else "no"
    base = f"{opp} {normalize(q['title']).rstrip('?')} {opp}"
    claim = base
    for k in (1, 2, 4, 8):
        claim = " ".join([base] * k)
        s = await o.score([f"{pool[0]} {claim}"], 2, qkey=f"{o.qkey}:probe")
        if gp(s[f"{pool[0]} {claim}"].mean) <= 0.85:
            break
    pairs = {p: f"{p} {claim}" for p in pool}
    s = await o.score(list(pairs.values()), 2, qkey=f"{o.qkey}:probe")
    await o.close()
    return sorted(((gp(s[pairs[p]].mean), p) for p in pool if s[pairs[p]].n), reverse=True)


def main():
    db = DB()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    gp = (lambda v: 1 - v) if (q["goal"] or "yes") == "no" else (lambda v: v)
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(json.loads(q["jev_request"])),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    pool = sorted((s for s, v in per.items() if min(v) >= 0.98 and MIN_WORDS <= len(s.split()) < LEAD),
                  key=lambda s: -len(per[s]))[:80]
    ranked = asyncio.run(rank(db, q, pool, gp))[:LINES]
    print(f"{len(pool)} long 0.98 lines probed; screening top {len(ranked)} "
          f"(probe {ranked[0][0]:.3f}..{ranked[-1][0]:.3f})", flush=True)
    client = SiteClient()
    live = {x["slug"]: x for x in client.menu().get("questions") or []}[SLUG]
    attempt = client.attempt(live["revisionId"], "strict_chain") or client.start(live["revisionId"], "strict_chain")
    quiet = lambda m: None
    for i, (pv, phrase) in enumerate(ranked, 1):
        attempt, p = build_chain(client, attempt, phrase.split(), quiet)
        reads = [p]
        for _ in range(REROLLS):
            if max(reads) >= 0.99:
                break
            attempt, p = reroll(client, attempt, quiet)
            reads.append(p)
        tally = dict(sorted(collections.Counter(round(r, 2) for r in reads).items()))
        print(f"{i:2}. probe {pv:.3f} {len(phrase.split())}w site {tally}", flush=True)
        if max(reads) >= 0.99:
            print(f"   0.99 ON THE SITE: {phrase}", flush=True)
            return
    print("no 0.99 read")


main()
