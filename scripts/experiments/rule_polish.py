"""The leader's method on our new rule-stack 0.98 lines: single-word appends and inserts from his vocabulary
(rule, no exceptions, always, every, never, not, only, digits) and the question's own words."""
import asyncio, collections, json, random, re, sys
from jevlab.db import DB
from jevlab.oracle import Oracle, question_key

BOARDS = {"should-ketchup-go-on-a-hot-dog": 44, "is-git-gud-useful-advice": 63, "do-we-live-in-a-simulation": 48,
          "will-you-accept-my-job-application": 68, "is-10-hot-dogs-too-many": 40}
PER_LINE = int(sys.argv[1]) if len(sys.argv) > 1 else 40
VOCAB = ("rule rules 1 2 3 4 5 every always only never not no exceptions exception logic puzzle true axiom premise "
         "means forever absolutely all each must strictly literally world text therefore thus answer").split()
RULE_WORDS = {"rule", "rules", "puzzle", "axiom", "premise", "exceptions", "logic"}
rng = random.Random(4)


async def main():
    db = DB()
    grand = collections.Counter()
    for slug, lead in BOARDS.items():
        q = dict(db.all("SELECT * FROM questions WHERE slug=?", (slug,))[0])
        req = json.loads(q["jev_request"]) if isinstance(q["jev_request"], str) else q["jev_request"]
        gp = (lambda v: v) if (q["goal"] or "yes") != "no" else (lambda v: 1 - v)
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        roots = [s for s, v in per.items() if min(v) >= 0.98 and len(s.split()) <= lead - 3
                 and len(RULE_WORDS & set(s.lower().split())) >= 2]
        title = [w for w in re.findall(r"[A-Za-z0-9]+", q["title"])]
        pool = VOCAB + title
        cands = {}
        for root in roots[:12]:
            ws = root.split()
            made = 0
            while made < PER_LINE:
                w = rng.choice(pool)
                g = len(ws) if rng.random() < 0.5 else rng.randrange(len(ws) + 1)
                p = " ".join(ws[:g] + [w] + ws[g:])
                if p not in per and p not in cands:
                    cands[p] = root
                made += 1
        if not cands:
            print(f"{slug}: no rule-stack roots at 0.98 yet"); continue
        o = Oracle(db, req)
        s = await o.score(list(cands), 1)
        dist = collections.Counter(round(gp(s[p].mean), 2) for p in cands if s[p].n)
        grand.update(dist)
        hits = [p for p in cands if s[p].n and round(gp(s[p].mean), 2) >= 0.99]
        print(f"## {slug}: {len(roots)} rule roots, {len(cands)} edits -> {sorted(dist.items(), reverse=True)[:4]}",
              flush=True)
        if hits:
            again = await o.score(hits, 5)
            for p in hits:
                print(f"   HIT {len(p.split())}w rerolls {[round(gp(x), 2) for x in again[p].samples]} :: {p}",
                      flush=True)
        await o.close()
    n = sum(grand.values())
    print(f"\nALL {n}: {sorted(grand.items(), reverse=True)[:5]} rate99={grand[0.99] / max(n, 1):.4f}")


asyncio.run(main())
