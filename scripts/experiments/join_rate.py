"""Rate of 0.99 among joins of two dissimilar top-level lines, per hard board."""
import asyncio, collections, json, random, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key

BOARDS = {"should-ketchup-go-on-a-hot-dog": 44, "do-you-love-me": 62, "will-you-accept-my-job-application": 68,
          "is-git-gud-useful-advice": 63, "point-break-is-the-greatest-movie-of-all-time": 60,
          "do-we-live-in-a-simulation": 48, "is-10-hot-dogs-too-many": 40, "can-you-pay-your-rent-in-monopoly-money": 40}
PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
rng = random.Random(5)


def words(s):
    return set(s.lower().split())


def jac(a, b):
    return len(a & b) / len(a | b)


async def main():
    db = DB()
    grand = collections.Counter()
    for slug, lead in BOARDS.items():
        rows = db.all("SELECT * FROM questions WHERE slug=?", (slug,))
        if not rows:
            print("missing", slug)
            continue
        q = dict(rows[0])
        req = json.loads(q["jev_request"]) if isinstance(q["jev_request"], str) else q["jev_request"]
        gp = (lambda v: v) if (q["goal"] or "yes") != "no" else (lambda v: 1 - v)
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        top = [s for s, v in per.items() if min(v) >= 0.98 and len(s.split()) <= lead - 4]
        near = [s for s, v in per.items() if min(v) >= 0.97 and len(s.split()) <= lead - 4]
        ws = {s: words(s) for s in near}
        cap = lead - 1
        seen, joins = set(), []
        tries = 0
        while len(joins) < PAIRS and tries < PAIRS * 200:
            tries += 1
            a = rng.choice(top)
            b = rng.choice(near)
            if a == b or len(a.split()) + len(b.split()) > cap or jac(ws[a], ws[b]) >= 0.3:
                continue
            p = f"{a} {b}" if rng.random() < 0.5 else f"{b} {a}"
            if p in seen or p in per:
                continue
            seen.add(p)
            joins.append(p)
        if not joins:
            print(f"{slug}: no dissimilar pairs fit in {cap} words ({len(top)} top lines)")
            continue
        o = Oracle(db, req)
        s = await o.score(joins, 1)
        dist = collections.Counter(round(gp(s[p].mean), 2) for p in joins if s[p].n)
        grand.update(dist)
        hits = [p for p in joins if s[p].n and round(gp(s[p].mean), 2) >= 0.99]
        print(f"## {slug} leader {lead}w: {len(top)} top, {len(near)} >=0.97; {len(joins)} joins -> "
              f"{sorted(dist.items(), reverse=True)[:5]}", flush=True)
        if hits:
            again = await o.score(hits, 5)
            for p in hits:
                print(f"   HIT {len(p.split())}w rerolls {[round(gp(x), 2) for x in again[p].samples]} :: {p}", flush=True)
        await o.close()
    n = sum(grand.values())
    print(f"\nALL {n} joins: {sorted(grand.items(), reverse=True)[:6]}  rate99={grand[0.99] / max(n, 1):.4f}")


asyncio.run(main())
