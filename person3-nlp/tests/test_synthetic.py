"""Large-scale check against generated cases whose GOLD labels are decided before the text."""
import pytest

from nutricheck_nlp.synthetic import generate_cases

SEEDS = [11, 22, 33, 44, 55]
PER_SEED = 1000


@pytest.mark.parametrize("seed", SEEDS)
def test_synthetic_100_percent(engine, seed):
    cases = generate_cases(PER_SEED, seed=seed)
    failures = []
    for c in cases:
        r = engine.analyze_text(c.text)
        got = {n["key"]: n["status"] for n in r["nutrients"] + r["findings"]}
        diff = {k: (v, got[k]) for k, v in c.expected.items() if got[k] != v}
        if diff:
            failures.append((c.text, diff))
    assert not failures, f"{len(failures)}/{PER_SEED} failed, first: {failures[0]}"


def test_generator_is_reproducible():
    assert [c.text for c in generate_cases(20, seed=7)] == [c.text for c in generate_cases(20, seed=7)]
