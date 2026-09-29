"""How does the live site score a long strict line? Build one of our long 0.98 lines word by word, logging the
site's score at every prefix, then re-roll the last word. The board keeps our best-ever row, so this can only help."""
import asyncio, collections, json, os, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key
from jevlab.rules import normalize
from jevlab.publish import build_chain, reroll
from jevlab.site import SiteClient

SLUG, LEAD = sys.argv[1], int(sys.argv[2])
REROLLS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MIN_WORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 45
MIN_READS = int(sys.argv[5]) if len(sys.argv) > 5 else 3


async def probe_best(db, q, pool, flip):
    """The line that best resists a claim for the other answer, i.e. sits highest inside the rounded level."""
    o = Oracle(db, json.loads(q["jev_request"]))
    gp = (lambda v: 1 - v) if flip else (lambda v: v)
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
    vals = sorted(((gp(s[pairs[p]].mean), p) for p in pool if s[pairs[p]].n), reverse=True)
    print(f"probe claim x{k}; probe values {vals[0][0]:.3f} (best) .. {vals[-1][0]:.3f}", flush=True)
    await o.close()
    return vals[0][1]


def main():
    db = DB()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    flip = (q["goal"] or "yes") == "no"
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(json.loads(q["jev_request"])),)):
        per[r["state"]].append(round(1 - r["noul"] if flip else r["noul"], 2))
    pool = [s for s, v in per.items() if len(v) >= MIN_READS and min(v) >= 0.98 and MIN_WORDS <= len(s.split()) < LEAD]
    if not pool:
        print("no 0.98 line with 3+ reads"); return
    pool = sorted(pool, key=lambda s: -len(per[s]))[:40]
    phrase = os.environ.get("PHRASE") or asyncio.run(probe_best(db, q, pool, flip))
    print(f"line {len(phrase.split())}w, local reads {per[phrase]}: {phrase}", flush=True)
    client = SiteClient()
    live = {x["slug"]: x for x in client.menu().get("questions") or []}[SLUG]
    attempt = client.attempt(live["revisionId"], "strict_chain") or client.start(live["revisionId"], "strict_chain")
    prefixes = []
    attempt, p = build_chain(client, attempt, phrase.split(),
                             lambda m: (prefixes.append(m), print(m, flush=True)))
    rolls = [p]
    for i in range(REROLLS):
        attempt, p = reroll(client, attempt, print)
        rolls.append(p)
        print(f"  reroll {i + 1}: {p:.2f}", flush=True)
        if p >= 0.99:
            break
    print(f"site reads of the full line: {collections.Counter(round(r, 2) for r in rolls)}")


main()
