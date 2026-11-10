"""Evidence for the pitch: NSGA-II vs equal-budget random search (hypervolume), + savings vs the catalogue baseline."""
import numpy as np, pandas as pd, json
from pymoo.indicators.hv import HV
from optifrugal.optimizer import Spec, run_nsga2, baseline_design, LensProblem, OBJECTIVES, finalize
from optifrugal.surrogate import load
from optifrugal.simulator import decode

models = load()
rows = []
for P, D in [(-6, 62), (-3, 60), (-9, 66), (2, 58), (4, 60)]:
    spec = Spec(P=P, D=D)
    front = run_nsga2(spec, models)
    prob = LensProblem(spec, models)
    rng = np.random.default_rng(0)
    rnd = finalize(prob.design_frame(rng.random((spec.pop * spec.gens, 5))), spec)
    cols = [(OBJECTIVES[o][0], OBJECTIVES[o][1]) for o in spec.objectives]
    def M(df): return np.column_stack([s * df[c].values for c, s in cols])
    allp = np.vstack([M(front), M(rnd)]); lo, hi = allp.min(0), allp.max(0)
    norm = lambda A: (A - lo) / (hi - lo + 1e-12)
    hv = HV(ref_point=np.ones(len(cols)) * 1.1)
    b = baseline_design(spec)
    rows.append(dict(P=P, D=D, nsga_front=len(front), rand_feasible=len(rnd), HV_nsga=hv(norm(M(front))), HV_random=hv(norm(M(rnd))),
                     cost_saving_pct=100 * (1 - front.cost.min() / b.cost[0]),
                     thin_saving_pct=100 * (1 - front.t_max.min() / b.t_max[0]),
                     co2_saving_pct=100 * (1 - front.co2.min() / b.co2[0])))
df = pd.DataFrame(rows).round(3)
print(df.to_string(index=False))
df.to_csv("data/benchmark.csv", index=False)
