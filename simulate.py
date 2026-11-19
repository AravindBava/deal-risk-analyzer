import numpy as np
import math
from returns import five_year_flows, total_profit

RISKS = {
    "rent": (-0.1, 0, 0.1),
    "vacancy": (0.02, 0.05, 0.12),
    "repairs": (0.05, 0.08, 0.15),
    "tax": (.9, 1.0, 1.25),
    "appreciation": (-0.03, 0.03, 0.08)
}

# Correlations between risks. Rows and columns follow this order:
RISK_ORDER = ["rent", "vacancy", "repairs", "tax", "appreciation"]
CORR = np.array([
    [1.0, -0.4, 0.0, 0.0, 0.5],
    [-0.4, 1.0, 0.2, 0.0, -0.3],
    [0.0, 0.2, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.4],
    [0.5, -0.3, 0.0, 0.4, 1.0],
])


def draw(rng, low, likely, high):
    if low == high:
        return likely
    return rng.triangular(low, likely, high)


def normal_to_pct(z):
    return 0.5 * (1 + math.erf(z/math.sqrt(2)))


def triangle_from_pct(u, low, likely, high):
    if low == high:
        return likely
    cut = (likely - low) / (high - low)
    if u < cut:
        return low + math.sqrt(u * (high - low) * (likely - low))
    return high - math.sqrt((1 - u) * (high - low) * (high - likely))


def draw_risks(rng, L, risks):
    # Five bell-curve numbers that move together the way CORR says
    z = L @ rng.standard_normal(5)
    draws = {}
    for i in range(5):
        name = RISK_ORDER[i]
        draws[name] = triangle_from_pct(normal_to_pct(z[i]), *risks[name])
    return draws


def run_simulation(deal, n=10000, seed=42, risks=RISKS, corr=CORR):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    profits = []
    first_year = []
    for _ in range(n):
        d = draw_risks(rng, L, risks)
        rent = d["rent"]
        vacancy = d["vacancy"] + (deal["vacancy"] - RISKS["vacancy"][1])
        repairs = d["repairs"] + (deal["repairs_pct"] - RISKS["repairs"][1])
        tax = d["tax"]
        appreciation = d["appreciation"]

        this_sim = dict(deal)
        this_sim["monthly_rent"] = deal["monthly_rent"] * (1 + rent)
        this_sim["vacancy"] = vacancy
        this_sim["repairs_pct"] = repairs
        this_sim["annual_tax"] = deal["annual_tax"] * tax

        flows = five_year_flows(this_sim, appreciation=appreciation)
        profits.append(total_profit(flows))
        first_year.append(flows[1])

    return np.array(profits), np.array(first_year)


def summarize(profits, first_year):
    results = dict()
    results["loss_chance"] = float((profits < 0).mean())
    results["median"] = float(np.median(profits))
    results["bad_case"] = float(np.percentile(profits, 5))
    results["good_case"] = float(np.percentile(profits, 95))
    results["neg_year1"] = float((first_year < 0).mean())

    return results
