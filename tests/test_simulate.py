import numpy as np
from pytest import approx
from simulate import run_simulation, normal_to_pct, triangle_from_pct, CORR, RISKS, draw_risks
from deal import TEST_DEAL


NORMAL_VAL = {
    "rent": (0, 0, 0),
    "vacancy": (0.05, 0.05, 0.05),
    "repairs": (0.08, 0.08, 0.08),
    "tax": (1.0, 1.0, 1.0),
    "appreciation": (0.03, 0.03, 0.03)
}


def test_run_simulation():
    profits, first_year = run_simulation(TEST_DEAL, n=5, risks=NORMAL_VAL)

    assert profits == approx(30874.79, abs=0.01)


def test_normal_to_pct():
    assert normal_to_pct(0) == approx(0.5)
    assert normal_to_pct(1.645) == approx(0.95, abs=0.001)


def test_triangle_ends_and_middle():
    assert triangle_from_pct(0, -0.1, 0, 0.1) == approx(-0.1)
    assert triangle_from_pct(1, -0.1, 0, 0.1) == approx(0.1)
    assert triangle_from_pct(0.5, -0.1, 0, 0.1) == approx(0)


def test_triangle_lopsided():
    assert triangle_from_pct(0.5, 0.02, 0.05, 0.12) == approx(0.0608, abs=0.0001)


def test_draws_follow_correlation():
    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(CORR)
    rents = []
    growth = []
    for _ in range(5000):
        d = draw_risks(rng, L, RISKS)
        rents.append(d["rent"])
        growth.append(d["appreciation"])
    assert np.corrcoef(rents, growth)[0, 1] == approx(0.5, abs=0.05)