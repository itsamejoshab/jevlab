"""Shrink confirmed 0.99 lines by dropping words while they hold, then queue the shortest in the vault."""
import asyncio, collections, json, sys
from jevlab import vault
from jevlab.db import DB
from jevlab.objective import Leader
from jevlab.oracle import Oracle, question_key

SLUG, LEAD, MODE = sys.argv[1], int(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "strict_chain")


async def main():
    db = DB()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    req = json.loads(q["jev_request"])
    flip = (q["goal"] or "yes") == "no"
    gp = (lambda v: 1 - v) if flip else (lambda v: v)
    o = Oracle(db, req)

    def confirmed():
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        return {s: v for s, v in per.items() if len(v) >= 5 and min(v) >= 0.99}

    for root in sorted(confirmed(), key=lambda s: len(s.split()))[:4]:
        line = root.split()
        while True:
            cands = [" ".join(line[:i] + line[i + 1:]) for i in range(len(line))]
            s = await o.score(cands, 1)
            ups = [p for p in cands if s[p].n and round(gp(s[p].mean), 2) >= 0.99]
            if not ups:
                break
            s = await o.score(ups, 5)
            ok = [p for p in ups if min(round(gp(x), 2) for x in s[p].samples) >= 0.99]
            if not ok:
                break
            line = ok[0].split()
            print(f"   {len(line)}w :: {ok[0]}", flush=True)
    leader = Leader(0.99, LEAD, "Chris Judd")
    for s, v in sorted(confirmed().items(), key=lambda kv: len(kv[0].split()))[:3]:
        if len(s.split()) >= LEAD:
            continue
        e = vault.save(SLUG, MODE, s, p_mean=sum(v) / len(v), p_lcb=min(v), spread=0.0, n=len(v),
                       units=len(s.split()), leader=leader, beats=True, title=q["title"], origin="scenario_join",
                       status="queued")
        print(f"queued {e['units']}w n={len(v)} :: {s}")
    await o.close()


asyncio.run(main())
