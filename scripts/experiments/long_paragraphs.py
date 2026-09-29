"""Long fluent paragraphs from strong LLMs: do they reach 0.99 under the leader's length?"""
import asyncio, collections, json, random, sys
from jevlab.db import DB
from jevlab.llm import LLM
from jevlab.oracle import Oracle, question_key
from jevlab.rules import RuleError, check_phrase, normalize

BOARDS = {"should-ketchup-go-on-a-hot-dog": 44, "do-you-love-me": 62, "will-you-accept-my-job-application": 68,
          "is-git-gud-useful-advice": 63, "point-break-is-the-greatest-movie-of-all-time": 60,
          "do-we-live-in-a-simulation": 48, "is-10-hot-dogs-too-many": 40, "should-we-nerf-whoever-is-winning": 66}
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["google/gemini-3.8-flash", "google/gemini-3.1-pro-preview"]
rng = random.Random(9)


def prompt(title, goal, lo, hi, examples):
    return (f"Question put to a judge model: {title!r}\nWe want the judge to answer {goal.upper()} with total "
            f"certainty. Write 12 different passages of {lo} to {hi} words each. Each passage is read by the judge "
            f"as the whole context before it answers. Make each one overwhelmingly, unambiguously point to "
            f"{goal.upper()}: vary the voice (first person, official ruling, story, rulebook, expert verdict, "
            f"dialogue, list of facts), restate the verdict more than once, stack reasons, no hedging, no irony, "
            f"no mention of the opposite answer being possible. Plain words, basic punctuation only.\n"
            f"Short lines that already score very high, for inspiration (do not just copy them):\n"
            + "\n".join(f"- {e}" for e in examples)
            + '\nReturn JSON: {"passages": ["...", ...]}')


async def main():
    db, llm = DB(), LLM()
    grand = collections.Counter()
    for slug, lead in BOARDS.items():
        rows = db.all("SELECT * FROM questions WHERE slug=?", (slug,))
        if not rows:
            print("missing", slug); continue
        q = dict(rows[0])
        req = json.loads(q["jev_request"]) if isinstance(q["jev_request"], str) else q["jev_request"]
        goal = q["goal"] or "yes"
        gp = (lambda v: v) if goal != "no" else (lambda v: 1 - v)
        per = collections.defaultdict(list)
        for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
            per[r["state"]].append(round(gp(r["noul"]), 2))
        top = sorted((s for s, v in per.items() if min(v) >= 0.98), key=lambda s: len(s.split()))
        examples = rng.sample(top, min(6, len(top)))
        cap = lead - 1
        passages = []
        for model in MODELS:
            for lo, hi in ((20, 35), (max(30, cap - 20), cap - 2)):
                try:
                    data = await llm.json(model, "Reply with JSON only.", prompt(q["title"], goal, lo, hi, examples),
                                          temperature=1.0, max_tokens=6000)
                except Exception as e:
                    print("  llm fail", model, e); continue
                for t in (data or {}).get("passages") or []:
                    try:
                        clean = check_phrase(normalize(str(t)), [], cap)
                    except RuleError:
                        continue
                    passages.append((model.split("/")[1], clean))
        cands = {}
        for m, p in passages:
            cands[p] = f"{m}:para"
            best = min(top, key=lambda s: len(s.split())) if top else None
            if best and len(best.split()) + len(p.split()) <= cap:
                cands[f"{p} {best}"] = f"{m}:para+best"
        o = Oracle(db, req)
        s = await o.score(list(cands), 1)
        res = collections.defaultdict(collections.Counter)
        hits = []
        for p, kind in cands.items():
            if s[p].n:
                v = round(gp(s[p].mean), 2)
                res[kind][v] += 1
                grand[v] += 1
                if v >= 0.99:
                    hits.append(p)
        print(f"## {slug} (0.99/{lead}w):", {k: sorted(c.items(), reverse=True)[:4] for k, c in res.items()}, flush=True)
        if hits:
            again = await o.score(hits, 5)
            for p in hits:
                print(f"   HIT {len(p.split())}w rerolls {[round(gp(x), 2) for x in again[p].samples]} :: {p}", flush=True)
        await o.close()
    n = sum(grand.values())
    print(f"\nALL {n}: {sorted(grand.items(), reverse=True)[:6]} rate99={grand[0.99] / max(n, 1):.4f}")


asyncio.run(main())
