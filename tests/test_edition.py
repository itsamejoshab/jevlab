"""Mirror editions: Laya and Clef follow the same paths as Kev."""

import os
import subprocess
import sys

import pytest

from jevlab.config import EDITIONS
from jevlab.site.client import SiteClient
from jevlab.transfer import import_from_jev


def test_editions_include_laya_and_clef():
    assert EDITIONS == ("jev", "kev", "laya", "clef")


def test_referer_follows_edition(tmp_path):
    missing = tmp_path / "no-session.json"
    jev = SiteClient("jev", session_path=missing)._headers()["referer"]
    kev = SiteClient("kev", session_path=missing)._headers()["referer"]
    laya = SiteClient("laya", session_path=missing)._headers()["referer"]
    clef = SiteClient("clef", session_path=missing)._headers()["referer"]
    assert jev.endswith("/") and not jev.endswith("/jev/")
    assert kev.endswith("/kev/")
    assert laya.endswith("/laya/")
    assert clef.endswith("/clef/")


def test_import_jev_refuses_the_jev_edition():
    with pytest.raises(RuntimeError, match="Kev, Laya, or Clef"):
        import_from_jev(None)


def test_laya_edition_defaults(tmp_path):
    env = os.environ.copy()
    env["JEV_EDITION"] = "laya"
    env["JEVLAB_DATA"] = str(tmp_path)
    for key in ("JEV_MODEL", "JEV_ORACLE_BACKENDS", "JEV_ORACLE_CONCURRENCY", "JEV_ORACLE_MAX_CONCURRENCY",
                "JEV_ORACLE_TIMEOUT", "JEV_TRIAGE_K", "JEV_LAYA_CACHE"):
        env.pop(key, None)
    script = (
        "from pathlib import Path\n"
        "from jevlab.config import (DATA, EDITION, JEV_MODEL, LAYA_CACHE, MIRROR, ORACLE_BACKENDS,\n"
        "                           ORACLE_CONCURRENCY, ORACLE_MAX_CONCURRENCY, ROOT, TRIAGE_K)\n"
        "assert EDITION == 'laya' and MIRROR\n"
        "assert DATA.name == 'laya'\n"
        "assert JEV_MODEL == 'convaiinnovations/laya'\n"
        "assert ORACLE_BACKENDS == ['huggingface']\n"
        "assert ORACLE_CONCURRENCY == 1 and ORACLE_MAX_CONCURRENCY == 1\n"
        "assert LAYA_CACHE == Path.home() / '.cache' / 'jevlab' / 'huggingface'\n"
        "assert ROOT not in LAYA_CACHE.parents\n"
        "assert TRIAGE_K == 40\n"
    )
    subprocess.check_call([sys.executable, "-c", script], env=env)


def test_clef_edition_defaults(tmp_path):
    env = os.environ.copy()
    env["JEV_EDITION"] = "clef"
    env["JEVLAB_DATA"] = str(tmp_path)
    for key in ("JEV_MODEL", "JEV_ORACLE_BACKENDS", "JEV_ORACLE_CONCURRENCY", "JEV_ORACLE_MAX_CONCURRENCY",
                "JEV_ORACLE_TIMEOUT", "JEV_TRIAGE_K", "JEV_CLEF_CACHE", "JEV_CLEF_REPO", "JEV_CLEF_QUANT",
                "JEV_CLEF_DEVICE"):
        env.pop(key, None)
    script = (
        "from pathlib import Path\n"
        "from jevlab.config import (CLEF_CACHE, CLEF_QUANT, CLEF_REPO, DATA, EDITION, JEV_MODEL, MIRROR,\n"
        "                           ORACLE_BACKENDS, ORACLE_CONCURRENCY, ORACLE_MAX_CONCURRENCY, ROOT,\n"
        "                           TRIAGE_K)\n"
        "assert EDITION == 'clef' and MIRROR\n"
        "assert DATA.name == 'clef'\n"
        "assert JEV_MODEL == 'clef-flash'\n"
        "assert CLEF_REPO == 'Cloudflare/clef-flash'\n"
        "assert CLEF_QUANT == 'auto'\n"
        "assert ORACLE_BACKENDS == ['huggingface']\n"
        "assert ORACLE_CONCURRENCY == 1 and ORACLE_MAX_CONCURRENCY == 1\n"
        "assert CLEF_CACHE == Path.home() / '.cache' / 'jevlab' / 'huggingface'\n"
        "assert ROOT not in CLEF_CACHE.parents\n"
        "assert TRIAGE_K == 40\n"
    )
    subprocess.check_call([sys.executable, "-c", script], env=env)
