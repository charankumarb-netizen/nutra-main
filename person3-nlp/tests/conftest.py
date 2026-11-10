import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from nutricheck_nlp import NutriCheckNLP  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    return NutriCheckNLP()


@pytest.fixture()
def statuses(engine):
    def _run(text, **kw):
        r = engine.analyze_text(text, **kw)
        return {n["key"]: n["status"] for n in r["nutrients"] + r["findings"]}, r
    return _run
