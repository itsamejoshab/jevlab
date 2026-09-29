"""Does a choice/score twin of the site's yes/no question give finer resolution near the top?
Sends the site's noul question plus a yes/no choice twin and a 2-level score twin in one request, for lines we
know roll 0.99 and lines that roll 0.98 on the same board."""
import asyncio, collections, json, random
from jevlab.db import DB
from jevlab.oracle import BackendPool, build_request, question_key

BOARDS = ["is-cereal-a-soup", "is-chicago-deep-dish-real-pizza", "is-the-cake-a-lie", "could-you-beat-a-goose-in-a-fight"]
rng = random.Random(2)


def twins(request):
    q = request["questions"]["q"]
    request["questions"]["c"] = {"type": "choice", "instructions": q["instructions"],
                                 "criteria": {"yes": None, "no": None}}
    request["questions"]["s"] = {"type": "score", "instructions": q["instructions"], "criteria": ["No", "Yes"]}
    return request


async def main():
    db = DB(readonly=True)
    pool = BackendPool(8)
    backend = next(b for b in pool.backends if b.name == "typesafe")
    for slug in BOARDS:
        q = dict(db.all("SELECT * FROM questions WHERE slug=?", (slug,))[0])
        base = json.loads(q["jev_request"])
        goal = q["goal"] or "yes"
        gp = (lambda v: v) if goal != "no" else (lambda v: 1 - v)
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(base),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        hi = [s for s, v in per.items() if len(v) >= 3 and min(v) >= 0.99]
        lo = [s for s, v in per.items() if len(v) >= 3 and all(x == 0.98 for x in v)]
        mid = [s for s, v in per.items() if len(v) >= 3 and all(x == 0.97 for x in v)]
        picks = [("0.99", s) for s in rng.sample(hi, min(6, len(hi)))] + \
                [("0.98", s) for s in rng.sample(lo, min(6, len(lo)))] + \
                [("0.97", s) for s in rng.sample(mid, min(4, len(mid)))]
        print(f"\n## {slug} (goal {goal})")

        async def one(label, state):
            r = await backend.http.post("/systemone", json=twins(build_request(base, state)))
            a = r.json().get("answers") or {}
            return label, a

        for label, a in await asyncio.gather(*(one(l, s) for l, s in picks)):
            c, s = a.get("c") or {}, a.get("s") or {}
            print(f"  site {label}  noul {a.get('q', {}).get('noul')}  choice {c.get('probabilities')} "
                  f"conf {c.get('confidence')}  score {s.get('score')} probs {s.get('probabilities')} conf {s.get('confidence')}")
    await pool.close()


asyncio.run(main())
