"""Did the oracle drift? Rescore lines that read 0.99 before a cutoff and compare fresh samples."""
import asyncio, collections, json, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key

CUTOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 1759000000
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 6


async def main():
    db = DB()
    old, new = [], []
    for q in db.all("SELECT * FROM questions"):
        req = json.loads(q["jev_request"])
        qk = question_key(req)
        flip = (q["goal"] or "yes") == "no"
        rows = db.all("SELECT state, noul, at FROM oracle_samples WHERE qkey=? AND at < ?", (qk, CUTOFF))
        per = collections.defaultdict(list)
        for r in rows:
            per[r["state"]].append(1 - r["noul"] if flip else r["noul"])
        tops = [s for s, v in per.items() if len(v) >= 2 and min(v) >= 0.985][:PER]
        if not tops:
            continue
        o = Oracle(db, req)
        have = {s: len(per[s]) + len(db.all("SELECT 1 FROM oracle_samples WHERE qkey=? AND state=? AND at >= ?",
                                           (qk, s, CUTOFF))) for s in tops}
        sc = await o.score(tops, max(have.values()) + 2)
        for s in tops:
            fresh = db.all("SELECT noul FROM oracle_samples WHERE qkey=? AND state=? AND at >= ?", (qk, s, CUTOFF))
            f = [round(1 - r["noul"] if flip else r["noul"], 2) for r in fresh]
            old.append(round(min(per[s]), 2))
            new.extend(f)
            print(q["slug"][:40], [round(x, 2) for x in per[s]], "->", f, flush=True)
        await o.close()
    print(f"lines {len(old)}  fresh samples {len(new)}  fresh at 0.99: {sum(x >= 0.99 for x in new)/max(len(new),1):.2f}")


asyncio.run(main())
