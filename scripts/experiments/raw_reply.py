"""Print one full /systemone reply (and the request shape) to see whether it carries more than 2 decimals."""
import asyncio, json
from jevlab.db import DB
from jevlab.oracle import BackendPool, build_request


async def main():
    db = DB(readonly=True)
    q = dict(db.all("SELECT * FROM questions WHERE slug='do-you-love-me'")[0])
    req = build_request(json.loads(q["jev_request"]), "engaged Of course will I sweetheart happily we hardly wait marry "
                        "darling adore and love you always much all my heart")
    print("REQUEST keys:", {k: (v if k != "state" else "...") for k, v in req.items()})
    pool = BackendPool(2)
    for backend in pool.backends:
        r = await backend.http.post("/systemone", json=req)
        print(f"\n{backend.name} HTTP {r.status_code} headers:", {k: v for k, v in r.headers.items()
                                                                if k.lower().startswith(("x-", "openrouter"))})
        print(r.text[:3000])
    await pool.close()


asyncio.run(main())
