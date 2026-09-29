"""Does a different scenario (who is speaking, what relationship, what is at stake) lift a tuned line past 0.98?
Each round an LLM writes scenario lines, shown the ones whose joins did best; each is joined with our best 0.98
lines in both orders, and any join reading 0.99 is rerolled."""
import asyncio, collections, json, random, sys
from jevlab.db import DB
from jevlab.llm import LLM
from jevlab.oracle import Oracle, question_key
from jevlab.rules import RuleError, check_phrase, normalize

SLUG = sys.argv[1]
LEAD = int(sys.argv[2])
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = random.Random(3)
MODELS = ["google/gemini-3.8-flash", "deepseek/deepseek-v4-flash"]


def prompt(title, answer, hint, best):
    shown = "".join(f"\n- ({v:.3f}) {p}" for v, p in best)
    more = (f"\nThe strongest so far (judge's average certainty in brackets); go further in the directions that "
            f"work, and also try completely new situations:{shown}") if best else ""
    return (f"A judge reads a short statement and answers the question \"{title}\" with {answer.upper()} or the "
            f"opposite. Write 30 statements that make {answer.upper()} as certain as possible. Each must use a "
            f"DIFFERENT speaker, relationship or situation (who is asking, who answers, what is at stake), "
            f"chosen so that {answer.upper()} is the only conceivable answer, e.g. {hint}. Raise the stakes: the "
            f"situation should make any other answer absurd or monstrous. Plain words only, "
            f"8 to 30 words each, no quotes.{more} JSON: {{\"texts\": [..]}}")


HINTS = {"do-you-love-me": "a mother to her newborn, a dog to its owner, wedding vows, a couple married 60 years",
         "will-you-accept-my-job-application": "the company owner is your mother, you are the only applicant, "
                                               "the job was created for you, the contract is already signed",
         "is-git-gud-useful-advice": "a beginner who got only a taunt, a teacher grading feedback, a harassment "
                                     "report, a style guide on constructive feedback"}


async def main():
    db, llm = DB(), LLM()
    q = dict(db.all("SELECT * FROM questions WHERE slug=?", (SLUG,))[0])
    req = json.loads(q["jev_request"])
    goal = q["goal"] or "yes"
    gp = (lambda v: v) if goal != "no" else (lambda v: 1 - v)
    per = collections.defaultdict(list)
    for r in db.all("SELECT state, noul FROM oracle_samples WHERE qkey=?", (question_key(req),)):
        per[r["state"]].append(round(gp(r["noul"]), 2))
    tops = sorted((s for s, v in per.items() if min(v) >= 0.98 and len(v) >= 2), key=lambda s: -len(per[s]))[:30]
    o = Oracle(db, req)
    rated, wins = {}, []
    for rnd in range(ROUNDS):
        best = sorted(((v, p) for p, v in rated.items()), reverse=True)[:8]
        got = await asyncio.gather(*(llm.json(m, "Reply with JSON only.",
                                              prompt(q["title"], goal, HINTS.get(SLUG, "a life or death emergency, a sworn expert, a court ruling, a child at risk"), best),
                                              temperature=1.0, max_tokens=6000) for m in MODELS for _ in range(2)),
                                   return_exceptions=True)
        texts = [t for g in got if isinstance(g, dict) for t in g.get("texts") or []]
        alone = {}
        for t in texts:
            try:
                alone[check_phrase(normalize(str(t)), [], 400)] = 1
            except RuleError:
                pass
        s = await o.score(list(alone), 1)
        vals = sorted(((round(gp(s[p].mean), 2), p) for p in alone if s[p].n), reverse=True)
        joins = collections.defaultdict(list)
        for v, p in vals[:25]:
            for t in rng.sample(tops, min(6, len(tops))):
                for j in (f"{p} {t}", f"{t} {p}"):
                    if len(j.split()) < LEAD:
                        joins[j].append(p)
        s = await o.score(list(joins), 1)
        by_scene = collections.defaultdict(list)
        jv = []
        for j, scenes in joins.items():
            if s[j].n:
                v = round(gp(s[j].mean), 2)
                jv.append((v, j))
                for p in scenes:
                    by_scene[p].append(v)
        for p, vs in by_scene.items():
            rated[p] = sum(vs) / len(vs)
        print(f"round {rnd}: {len(vals)} scenes, alone top {vals[0][0] if vals else None}, joins "
              f"{collections.Counter(v for v, _ in jv).most_common(4)}, calls {o.calls}", flush=True)
        for v, p in sorted(((v, p) for p, v in rated.items()), reverse=True)[:3]:
            print(f"   scene {v:.3f} :: {p}", flush=True)
        ups = [j for v, j in jv if v >= 0.99]
        if ups:
            again = await o.score(ups, 5)
            for j in ups:
                rolls = [round(gp(x), 2) for x in again[j].samples]
                print("   UP", rolls, len(j.split()), "w ::", j, flush=True)
                if min(rolls) >= 0.99:
                    wins.append(j)
    print(f"confirmed wins: {len(wins)}")
    await o.close()


asyncio.run(main())
