"""Do longer lines built from our best 0.98 lines reach 0.99? Joins, repeats and stacks, within the leader's length."""
import asyncio, collections, json, random
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key

BOARDS = {"would-you-let-copilot-name-your-variables": 41, "should-ketchup-go-on-a-hot-dog": 44, "do-you-love-me": 62,
          "will-you-accept-my-job-application": 68, "is-git-gud-useful-advice": 63,
          "point-break-is-the-greatest-movie-of-all-time": 60, "do-we-live-in-a-simulation": 48}
rng = random.Random(11)


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b)


async def main():
    db = DB()
    total = collections.Counter()
    for slug, lead in BOARDS.items():
        q = dict(db.all("SELECT * FROM questions WHERE slug=?", (slug,))[0])
        req = json.loads(q["jev_request"]) if isinstance(q["jev_request"], str) else q["jev_request"]
        gp = (lambda v: v) if (q["goal"] or "yes") != "no" else (lambda v: 1 - v)
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        tied = [s for s, v in per.items() if min(v) >= 0.98]
        # a diverse set: greedy pick of tied lines far from each other
        rng.shuffle(tied)
        div = []
        for s in sorted(tied, key=len):
            if all(jaccard(s, d) < 0.5 for d in div):
                div.append(s)
        for s in sorted(tied, key=len):
            if len(div) >= 12:
                break
            if s not in div and all(jaccard(s, d) < 0.75 for d in div):
                div.append(s)
        div = div[:12]
        if len(div) < 3:
            div = (div + rng.sample(tied, min(12, len(tied))))[:12]
        cap = lead - 1
        cands = {}
        for a in div:
            if 2 * len(a.split()) <= cap:
                cands[f"{a} {a}"] = "repeat"
        for _ in range(80):
            a, b = rng.sample(div, 2)
            if len(a.split()) + len(b.split()) <= cap:
                cands.setdefault(f"{a} {b}", "join")
        for _ in range(40):
            k = rng.sample(div, 3)
            if sum(len(x.split()) for x in k) <= cap:
                cands.setdefault(" ".join(k), "stack3")
        cands = dict(list(cands.items())[:45])
        oracle = Oracle(db, req)
        s = await oracle.score(list(cands), 1)
        res = collections.defaultdict(list)
        for p, kind in cands.items():
            if s[p].n:
                res[kind].append((round(gp(s[p].mean), 2), len(p.split()), p))
        print(f"\n## {slug} (leader 0.99/{lead}w, {len(tied)} tied 0.98 lines, {len(div)} diverse)")
        for kind, rows in res.items():
            dist = collections.Counter(r[0] for r in rows)
            total.update({f"{kind}:{k}": v for k, v in dist.items()})
            print(f"  {kind:7} n={len(rows):3} {sorted(dist.items(), reverse=True)}")
            for r in sorted(rows, reverse=True)[:1]:
                print(f"     best {r[0]} {r[1]}w {r[2] if r[0] >= 0.99 else r[2][:120]}")
            wins = [r for r in rows if r[0] >= 0.99 and r[1] < lead]
            if wins:
                again = await oracle.score([r[2] for r in wins], 5)
                for r in wins:
                    print(f"     WIN? {r[1]}w rerolls {[round(gp(x), 2) for x in again[r[2]].samples]}")
        await oracle.close()
    print("\nTOTAL", sorted(total.items()))


asyncio.run(main())
