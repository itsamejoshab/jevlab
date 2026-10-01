"""Mirror editions: Laya follows the same paths as Kev."""

import os
import subprocess
import sys

import pytest

from jevlab.config import EDITIONS
from jevlab.site.client import SiteClient
from jevlab.transfer import import_from_jev


def test_editions_include_laya():
    assert EDITIONS == ("jev", "kev", "laya")


def test_referer_follows_edition(tmp_path):
    missing = tmp_path / "no-session.json"
    jev = SiteClient("jev", session_path=missing)._headers()["referer"]
    kev = SiteClient("kev", session_path=missing)._headers()["referer"]
    laya = SiteClient("laya", session_path=missing)._headers()["referer"]
    assert jev.endswith("/") and not jev.endswith("/jev/")
    assert kev.endswith("/kev/")
    assert laya.endswith("/laya/")


def test_import_jev_refuses_the_jev_edition():
    with pytest.raises(RuntimeError, match="Kev or Laya"):
        import_from_jev(None)


def test_laya_edition_defaults(tmp_path):
    env = os.environ.copy()
    env["JEV_EDITION"] = "laya"
    env["JEVLAB_DATA"] = str(tmp_path)
    for key in ("JEV_MODEL", "JEV_ORACLE_BACKENDS", "JEV_ORACLE_CONCURRENCY", "JEV_ORACLE_MAX_CONCURRENCY",
                "JEV_ORACLE_TIMEOUT", "JEV_TRIAGE_K"):
        env.pop(key, None)
    script = (
        "from jevlab.config import DATA, EDITION, JEV_MODEL, MIRROR, ORACLE_BACKENDS, TRIAGE_K\n"
        "assert EDITION == 'laya' and MIRROR\n"
        "assert DATA.name == 'laya'\n"
        "assert JEV_MODEL == 'convaiinnovations/laya'\n"
        "assert ORACLE_BACKENDS == ['openrouter']\n"
        "assert TRIAGE_K == 40\n"
    )
    subprocess.check_call([sys.executable, "-c", script], env=env)
