"""
Physics-informed simulator for a single-vision spectacle lens (the demo product).

It stands in for the expensive tools ZEISS engineers would normally use (ray tracing, FEA,
costing sheets). All material / cost / CO2 numbers are ILLUSTRATIVE placeholders - swap the
tables below (or the whole dataset) for real ZEISS engineering data.

Design variables : material, front-curve (base curve) F1 [D], centre thickness tc [mm],
                   aspheric surface (0/1), coating package (0-3)
Requirements     : prescription power P [D], lens diameter D [mm]
Outputs          : cost [INR], weight [g], max thickness [mm], optical quality [0-100],
                   impact index [1.0 = pass], CO2e [kg], circularity [0-1], durability [0-100]
"""
import numpy as np
import pandas as pd

MATERIALS = pd.DataFrame(
    [
        # name,                 n,     abbe, dens, impact_f, cost/g, co2/kg, recyclable, hardness
        # n / abbe / density: published datasheets (see data/SOURCES.md). impact_f = 0.60 * (impact strength in-lbf / 75),
        # i.e. normalised so CR-39 = 0.60 (Hoya/PPG white paper: CR-39 75, 1.60 -> 89, 1.67 -> 97, PC and Trivex > 340).
        # cost/g (retail-price-ratio proxy), co2/kg, recyclable, hardness and the 1.56 / 1.74 impact values are ASSUMPTIONS.
        ("CR-39 (1.50)",         1.498, 58, 1.32, 0.60, 1.2, 3.8, 0.15, 0.80),
        ("Mid-index (1.56)",     1.560, 35, 1.22, 0.65, 1.7, 4.2, 0.15, 0.75),
        ("Trivex (1.53)",        1.530, 44, 1.11, 2.72, 3.4, 5.0, 0.20, 0.70),
        ("Polycarbonate (1.59)", 1.590, 30, 1.21, 2.72, 2.2, 6.5, 0.70, 0.45),
        ("MR-8 (1.60)",          1.600, 41, 1.30, 0.71, 4.4, 6.0, 0.10, 0.70),
        ("MR-7 (1.67)",          1.670, 31, 1.35, 0.78, 6.4, 6.8, 0.10, 0.65),
        ("MR-174 (1.74)",        1.740, 32, 1.47, 0.70, 9.6, 7.5, 0.10, 0.60),
    ],
    columns=["name", "n", "abbe", "density", "impact_f", "cost_pg", "co2_pkg", "recyclable", "hardness"],
)
COATINGS = pd.DataFrame(
    [
        # name,                          cost INR, transmission, co2 kg, durability bonus, blue-cut
        ("Hard coat only",                 12, None,  0.010, 0.00, 0),
        ("Hard coat + Anti-reflective",    38, 0.990, 0.030, 0.00, 0),
        ("HC + AR + Hydrophobic",          55, 0.990, 0.040, 0.30, 0),
        ("HC + AR + Hydrophobic + Blue-cut", 78, 0.965, 0.055, 0.30, 1),
    ],
    columns=["name", "cost", "trans", "co2", "dur_bonus", "blue_cut"],
)
N_MAT, N_COAT = len(MATERIALS), len(COATINGS)
IMPACT_REQ = 0.60          # impact index that equals CR-39 @ 2.0 mm (drop-ball-passing reference)
TE_MIN = 1.0               # minimum edge thickness for mounting [mm]
F1_RANGE, TC_RANGE = (0.5, 9.5), (1.0, 8.0)   # 1.0 mm = practical minimum centre thickness

# Default "catalogue" lens the optimiser is compared with (editable in the app)
BASELINE = dict(material=4, F1=4.0, tc=2.0, asph=0, coat=3)   # MR-8, spherical, stock 4 D curve, full coating


def _sag(F, n, h):
    """Signed sagitta of a refracting surface of power F [D] at semi-height h [mm]. Also returns validity."""
    Fa = np.maximum(np.abs(F), 1e-3)
    R = (n - 1.0) / Fa * 1000.0
    ok = R > h * 1.05
    Rs = np.where(ok, R, h * 1.05)
    return np.sign(F) * h * h / (Rs + np.sqrt(Rs * Rs - h * h)), ok


def edge_thickness(P, D, F1, tc, n):
    """Cheap analytic geometry: edge thickness [mm] and geometric validity."""
    h = D / 2.0
    s1, ok1 = _sag(F1, n, h)
    s2, ok2 = _sag(P - F1, n, h)
    return tc - s1 - s2, ok1 & ok2


def simulate(df: pd.DataFrame, noise: float = 0.0, rng=None) -> pd.DataFrame:
    """Vectorised ground-truth simulator. df needs columns P, D, material, F1, tc, asph, coat."""
    m = MATERIALS.iloc[df["material"].astype(int).values].reset_index(drop=True)
    c = COATINGS.iloc[df["coat"].astype(int).values].reset_index(drop=True)
    P, D, F1, tc, asph = (df[k].values.astype(float) for k in ("P", "D", "F1", "tc", "asph"))
    n = m["n"].values
    h = D / 2.0

    # --- geometry & volume (numerical integration of thickness profile) ---
    r = np.linspace(0, 1, 33)[None, :] * h[:, None]
    s1, ok1 = _sag(F1[:, None], n[:, None], r)
    s2, ok2 = _sag((P - F1)[:, None], n[:, None], r)
    t = np.clip(tc[:, None] - s1 - s2, 0, None)
    valid = (_sag(F1, n, h)[1]) & (_sag(P - F1, n, h)[1])
    vol_cm3 = np.trapezoid(2 * np.pi * r * t, r, axis=1) / 1000.0
    mass = vol_cm3 * m["density"].values
    t_max, t_min = t.max(axis=1), t.min(axis=1)
    te = tc - _sag(F1, n, h)[0] - _sag(P - F1, n, h)[0]

    # --- cost [INR] ---
    cost = mass * m["cost_pg"].values * 1.3 + 30 + 14 * asph + 1.8 * t_max + c["cost"].values
    # --- optical quality ---
    tca = 2.0 * np.abs(P) / m["abbe"].values                     # transverse chromatic aberration (prism dioptres)
    q_chrom = np.exp(-tca / 0.5)
    F_best = np.clip(5.5 + 0.55 * P, 0.5, 9.5)                    # Tscherning-like best-form base curve
    e_eff = np.abs(F1 - F_best) * (1 - 0.6 * asph)
    q_aber = np.exp(-0.15 * e_eff ** 2 - 0.004 * P ** 2 * (1 - 0.5 * asph))
    R_s = ((n - 1) / (n + 1)) ** 2
    trans = np.where(c["trans"].isna(), (1 - R_s) ** 2 * 0.99, c["trans"].fillna(0).values)
    oq = 100 * q_aber ** 0.40 * q_chrom ** 0.35 * trans ** 0.25
    # --- safety, sustainability, durability ---
    impact = m["impact_f"].values * (np.clip(t_min, 0.05, None) / 2.0) ** 0.8
    co2 = mass / 1000 * m["co2_pkg"].values * 1.4 + c["co2"].values + 0.004 * asph
    circ = m["recyclable"].values * (1 - 0.15 * (df["coat"].values >= 1)) * (1 - 0.15 * (df["coat"].values >= 2))
    dur = 100 * np.clip(m["hardness"].values * (1 + c["dur_bonus"].values), 0, 1)

    out = pd.DataFrame(dict(cost=cost, weight=mass, t_max=t_max, t_min=t_min, te=te, oq=oq, impact=impact,
                            co2=co2, circ=circ, dur=dur, valid=valid))
    if noise > 0:  # manufacturing / measurement variability
        rng = rng or np.random.default_rng(0)
        for k, s in dict(cost=1.0, weight=0.5, t_max=0.5, oq=1.0, impact=1.5, co2=1.0).items():
            out[k] = out[k] * np.exp(rng.normal(0, noise * s, len(out)))
    return out


def decode(x, allowed_mat, allowed_coat):
    """Map a [0,1]^5 genome to concrete design variables."""
    x = np.atleast_2d(x)
    am, ac = np.asarray(allowed_mat), np.asarray(allowed_coat)
    mi = np.minimum((x[:, 0] * len(am)).astype(int), len(am) - 1)
    ci = np.minimum((x[:, 4] * len(ac)).astype(int), len(ac) - 1)
    return pd.DataFrame(dict(
        material=am[mi], F1=F1_RANGE[0] + x[:, 1] * (F1_RANGE[1] - F1_RANGE[0]),
        tc=TC_RANGE[0] + x[:, 2] * (TC_RANGE[1] - TC_RANGE[0]),
        asph=(x[:, 3] > 0.5).astype(int), coat=ac[ci]))


def describe(row) -> str:
    return (f"{MATERIALS.name[int(row['material'])]}, base curve {row['F1']:.2f} D, centre thickness {row['tc']:.2f} mm, "
            f"{'aspheric' if int(row['asph']) else 'spherical'}, {COATINGS.name[int(row['coat'])]}")
