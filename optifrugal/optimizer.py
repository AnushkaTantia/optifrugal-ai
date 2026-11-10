"""Multi-objective optimisation (NSGA-II) on top of the ML surrogates, with simulator validation."""
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from .simulator import simulate, decode, edge_thickness, MATERIALS, N_MAT, N_COAT, IMPACT_REQ, TE_MIN, BASELINE
from .surrogate import predict, featurize

# objective name -> (column, sign: +1 minimise, -1 maximise)
OBJECTIVES = {"Cost": ("cost", 1), "Thickness": ("t_max", 1), "Weight": ("weight", 1),
              "Optical quality": ("oq", -1), "CO2e": ("co2", 1)}


@dataclass
class Spec:
    P: float = -4.0
    D: float = 62.0
    objectives: tuple = ("Cost", "Thickness", "Optical quality", "CO2e")
    oq_min: float = 65.0
    impact_margin: float = 1.0          # required impact index = IMPACT_REQ * margin
    t_max_cap: float = 14.0
    weight_cap: float = 40.0
    allowed_mat: tuple = tuple(range(N_MAT))
    allowed_coat: tuple = tuple(range(N_COAT))
    pop: int = 160
    gens: int = 70
    seed: int = 1


class LensProblem(Problem):
    def __init__(self, spec: Spec, models):
        super().__init__(n_var=5, n_obj=len(spec.objectives), n_ieq_constr=6, xl=0.0, xu=1.0)
        self.spec, self.models = spec, models

    def design_frame(self, x):
        d = decode(x, self.spec.allowed_mat, self.spec.allowed_coat)
        d["P"], d["D"] = self.spec.P, self.spec.D
        return d

    def _evaluate(self, x, out, *a, **k):
        s, d = self.spec, self.design_frame(x)
        pr = predict(self.models, d)
        n = MATERIALS.n.values[d.material.values]
        te, ok = edge_thickness(d.P.values, d.D.values, d.F1.values, d.tc.values, n)
        out["F"] = np.column_stack([OBJECTIVES[o][1] * pr[OBJECTIVES[o][0]].values for o in s.objectives])
        out["G"] = np.column_stack([IMPACT_REQ * s.impact_margin - pr.impact.values, s.oq_min - pr.oq.values,
                                    TE_MIN - te, pr.t_max.values - s.t_max_cap, pr.weight.values - s.weight_cap,
                                    (~ok).astype(float)])


def run_nsga2(spec: Spec, models) -> pd.DataFrame:
    prob = LensProblem(spec, models)
    res = minimize(prob, NSGA2(pop_size=spec.pop, eliminate_duplicates=True), get_termination("n_gen", spec.gens),
                   seed=spec.seed, verbose=False)
    if res.X is None:
        return pd.DataFrame()
    return finalize(prob.design_frame(res.X), spec)


def finalize(designs: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Re-score designs with the ground-truth simulator and keep unique, feasible ones."""
    d = designs.copy()
    d["P"], d["D"] = spec.P, spec.D
    d = d.assign(F1=d.F1.round(2), tc=d.tc.round(2)).drop_duplicates(["material", "F1", "tc", "asph", "coat"]).reset_index(drop=True)
    sim = simulate(d)
    sur = predict_cols(d)
    out = pd.concat([d, sim, sur], axis=1)
    feas = (out.impact >= IMPACT_REQ * spec.impact_margin - 1e-9) & (out.oq >= spec.oq_min - 1e-6) & (out.te >= TE_MIN - 1e-6) \
           & (out.t_max <= spec.t_max_cap + 1e-6) & (out.weight <= spec.weight_cap + 1e-6) & out.valid
    return out[feas].reset_index(drop=True)


def predict_cols(d):
    from .surrogate import load
    p = predict(_MODELS_CACHE.setdefault("m", load()), d)
    return p.add_prefix("sur_")


_MODELS_CACHE = {}


def baseline_design(spec: Spec, base=None) -> pd.DataFrame:
    b = dict(BASELINE, **(base or {}))
    d = pd.DataFrame([dict(b, P=spec.P, D=spec.D, asph=int(b["asph"]))])
    # a conventional catalogue lens uses the thickness the standard process needs: edge >= TE_MIN for plus lenses
    te = edge_thickness(d.P.values, d.D.values, d.F1.values, d.tc.values, MATERIALS.n.values[d.material.values])[0]
    d.loc[:, "tc"] = np.maximum(d.tc, d.tc + (TE_MIN - te))
    return pd.concat([d, simulate(d)], axis=1)


def recommend(front: pd.DataFrame, weights: dict) -> int:
    """TOPSIS-style pick: closest normalised point to the ideal under customer-need weights."""
    cols = {"cost": 1, "t_max": 1, "weight": 1, "oq": -1, "co2": 1, "dur": -1, "circ": -1}
    Z = []
    for c, s in cols.items():
        v = front[c].values * s
        rng = v.max() - v.min()
        Z.append((v - v.min()) / rng if rng > 1e-12 else np.zeros(len(v)))
    Z = np.array(Z).T
    w = np.array([weights.get(c, 0.0) for c in cols])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    return int(np.argmin(np.sqrt((w * Z ** 2).sum(axis=1))))   # weighted distance to the ideal point


# ---------------------------------------------------------------- customer needs -> objective weights / constraints
NEEDS = ["Affordability", "Thin & light", "Optical clarity", "Safety & durability", "Eco-friendliness", "Screen-use comfort"]
PERSONAS = {
    "Rural first-time wearer (budget)": [5, 2, 3, 3, 2, 1],
    "Student / heavy screen use":       [3, 3, 4, 3, 2, 5],
    "Child / active wearer":            [3, 2, 3, 5, 2, 2],
    "Fashion-focused high-power":       [2, 5, 4, 3, 2, 3],
    "Eco-conscious professional":       [2, 3, 4, 3, 5, 3],
}


def needs_to_weights(needs: dict) -> dict:
    a, tl, c, sd, eco, scr = (needs[k] for k in NEEDS)
    return dict(cost=a, t_max=tl * .6, weight=tl * .4, oq=c, dur=sd * .5, co2=eco * .6, circ=eco * .4)


def needs_to_constraints(needs: dict) -> dict:
    return dict(impact_margin=1.0 + 0.2 * max(0, needs["Safety & durability"] - 3),
                oq_min=45 + 8 * needs["Optical clarity"],
                coat=(3,) if needs["Screen-use comfort"] >= 4 else None)


# ---------------------------------------------------------------- feature value-for-money
def feature_value(design: pd.Series, spec: Spec) -> pd.DataFrame:
    """Marginal cost / benefit of each optional feature, relative to the same lens without it."""
    base = pd.DataFrame([design[["material", "F1", "tc", "asph", "coat", "P", "D"]].to_dict()])
    rows = []
    def delta(name, mod, benefit):
        a = base.copy(); b = base.copy()
        for k, v in mod.items(): b[k] = v
        ra, rb = simulate(a).iloc[0], simulate(b).iloc[0]
        rows.append(dict(Feature=name, ExtraCostINR=rb.cost - ra.cost, QualityGain=rb.oq - ra.oq,
                         ThicknessChangeMM=rb.t_max - ra.t_max, Benefit=benefit))
    # features toggled relative to the *same design without the feature*
    delta("Aspheric surface", dict(asph=1 - int(design.asph)) if int(design.asph) == 0 else dict(asph=0), "Flatter, thinner lens; less peripheral blur")
    delta("Anti-reflective coating", dict(coat=1) if design.coat == 0 else dict(coat=0), "Higher transmission, less glare")
    delta("Hydrophobic layer", dict(coat=2) if design.coat < 2 else dict(coat=1), "Cleanability, scratch resistance")
    delta("Blue-cut layer", dict(coat=3) if design.coat != 3 else dict(coat=2), "Screen-use comfort preference")
    return pd.DataFrame(rows)


def material_frontier(spec: Spec, base_design=None, grid=14) -> pd.DataFrame:
    """Best feasible design per material (cheapest, and thinnest) found by dense grid on the true simulator."""
    F1 = np.linspace(0.5, 9.5, 24); TC = np.linspace(0.8, 6.0, 14)
    rows = []
    for m in spec.allowed_mat:
        g = pd.MultiIndex.from_product([[m], F1, TC, [0, 1], list(spec.allowed_coat)], names=["material", "F1", "tc", "asph", "coat"]).to_frame(index=False)
        g["P"], g["D"] = spec.P, spec.D
        s = pd.concat([g, simulate(g)], axis=1)
        s = s[(s.impact >= IMPACT_REQ * spec.impact_margin) & (s.oq >= spec.oq_min) & (s.te >= TE_MIN) & (s.t_max <= spec.t_max_cap) & s.valid & (s.weight <= spec.weight_cap)]
        if len(s) == 0:
            rows.append(dict(material=m, name=MATERIALS.name[m], feasible=False)); continue
        c, t = s.loc[s.cost.idxmin()], s.loc[s.t_max.idxmin()]
        rows.append(dict(material=m, name=MATERIALS.name[m], feasible=True, min_cost=c.cost, cost_at_min_cost_t=c.t_max,
                         min_thickness=t.t_max, cost_at_min_thickness=t.cost, best_weight=s.weight.min(), best_oq=s.oq.max(), best_co2=s.co2.min()))
    return pd.DataFrame(rows)
