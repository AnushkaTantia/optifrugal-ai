"""ML surrogates: gradient-boosted regressors trained on simulated design data."""
import json, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from .simulator import MATERIALS, simulate, edge_thickness, F1_RANGE, TC_RANGE

TARGETS = ["cost", "weight", "t_max", "oq", "impact", "co2"]
FEATURES = ["P", "D", "F1", "tc", "asph", "coat", "n", "abbe", "density", "impact_f", "cost_pg", "co2_pkg", "te"]


def make_dataset(n=40000, seed=7, noise=0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        P=rng.uniform(-10, 6, n).round(2), D=rng.uniform(50, 75, n).round(1),
        material=rng.integers(0, len(MATERIALS), n), F1=rng.uniform(*F1_RANGE, n),
        tc=rng.uniform(*TC_RANGE, n), asph=rng.integers(0, 2, n), coat=rng.integers(0, 4, n)))
    out = simulate(df, noise=noise, rng=rng)
    df = pd.concat([df, out], axis=1)
    return df[df.valid].reset_index(drop=True)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    m = MATERIALS.iloc[df["material"].astype(int).values].reset_index(drop=True)
    X = pd.DataFrame({k: df[k].values for k in ["P", "D", "F1", "tc", "asph", "coat"]})
    for k in ["n", "abbe", "density", "impact_f", "cost_pg", "co2_pkg"]:
        X[k] = m[k].values
    X["te"] = edge_thickness(X.P.values, X.D.values, X.F1.values, X.tc.values, X.n.values)[0]
    return X[FEATURES]


def train(df, path="models/surrogates.joblib", metrics_path="models/metrics.json"):
    X, tr = featurize(df), None
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=1)
    models, metrics = {}, {}
    for t in TARGETS:
        mdl = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=48,
                                            l2_regularization=0.1, random_state=0)
        mdl.fit(X.iloc[idx_tr], df[t].iloc[idx_tr])
        pred = mdl.predict(X.iloc[idx_te])
        metrics[t] = dict(r2=float(r2_score(df[t].iloc[idx_te], pred)),
                          mape_pct=float(100 * mean_absolute_percentage_error(df[t].iloc[idx_te], pred)))
        models[t] = mdl
    joblib.dump(models, path)
    json.dump(dict(metrics=metrics, n_train=len(idx_tr), n_test=len(idx_te)), open(metrics_path, "w"), indent=2)
    return models, metrics


def load(path="models/surrogates.joblib"):
    return joblib.load(path)


def predict(models, df: pd.DataFrame) -> pd.DataFrame:
    X = featurize(df)
    return pd.DataFrame({t: models[t].predict(X) for t in TARGETS})
