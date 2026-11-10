import json, os
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st
from optifrugal.simulator import MATERIALS, COATINGS, BASELINE, describe, IMPACT_REQ, simulate
from optifrugal.surrogate import load, predict
from optifrugal.optimizer import (Spec, run_nsga2, baseline_design, recommend, feature_value, material_frontier,
                                  NEEDS, PERSONAS, needs_to_weights, needs_to_constraints, OBJECTIVES)

if not os.path.exists("models/surrogates.joblib"):   # first run after a fresh clone: build data + models (about 10 s)
    from optifrugal.surrogate import make_dataset, train
    os.makedirs("models", exist_ok=True); os.makedirs("data", exist_ok=True)
    _df = make_dataset(); _df.to_csv("data/lens_design_dataset.csv", index=False); train(_df)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
st.set_page_config(page_title="OptiFrugal AI", layout="wide")

@st.cache_resource
def models(): return load()

@st.cache_data(show_spinner="Running NSGA-II on the surrogate models...")
def optimise(P, D, objectives, oq_min, margin, tcap, wcap, mats, coats, pop, gens, seed):
    return run_nsga2(Spec(P, D, tuple(objectives), oq_min, margin, tcap, wcap, tuple(mats), tuple(coats), pop, gens, seed), models())

@st.cache_data
def mat_front(P, D, oq_min, margin, tcap, wcap, mats, coats):
    return material_frontier(Spec(P, D, ("Cost",), oq_min, margin, tcap, wcap, tuple(mats), tuple(coats)))

def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig

# ------------------------------------------------------------------ sidebar
st.sidebar.title("OptiFrugal AI")
st.sidebar.caption("Design requirements")
P = st.sidebar.slider("Prescription power (D)", -10.0, 6.0, -4.0, 0.25)
D = st.sidebar.slider("Lens diameter needed (mm)", 50, 75, 62)
st.sidebar.caption("Customer needs  (feature prioritisation)")
persona = st.sidebar.selectbox("Customer persona", list(PERSONAS))
needs = {n: st.sidebar.slider(n, 1, 5, PERSONAS[persona][i], key=f"{persona}-{n}") for i, n in enumerate(NEEDS)}
cons, weights = needs_to_constraints(needs), needs_to_weights(needs)
with st.sidebar.expander("Engineering constraints"):
    oq_min = st.slider("Minimum optical quality", 30, 90, int(cons["oq_min"]))
    margin = st.slider("Impact safety margin (x pass level)", 1.0, 2.0, float(cons["impact_margin"]), 0.05)
    tcap = st.slider("Max thickness (mm)", 4.0, 20.0, 14.0)
    wcap = st.slider("Max weight (g)", 5.0, 60.0, 40.0)
    mats = st.multiselect("Allowed materials", list(range(len(MATERIALS))), default=list(range(len(MATERIALS))), format_func=lambda i: MATERIALS.name[i])
    coat_default = list(cons["coat"]) if cons["coat"] else list(range(4))
    coats = st.multiselect("Allowed coatings", list(range(4)), default=coat_default, format_func=lambda i: COATINGS.name[i])
with st.sidebar.expander("Baseline = conventional catalogue lens"):
    b_mat = st.selectbox("Material", list(range(len(MATERIALS))), index=BASELINE["material"], format_func=lambda i: MATERIALS.name[i])
    b_F1 = st.slider("Base curve (D)", 0.5, 9.5, BASELINE["F1"], 0.5)
    b_tc = st.slider("Centre thickness (mm)", 1.0, 6.0, BASELINE["tc"], 0.1)
    b_asph = st.checkbox("Aspheric", bool(BASELINE["asph"]))
    b_coat = st.selectbox("Coating", list(range(4)), index=BASELINE["coat"], format_func=lambda i: COATINGS.name[i])
with st.sidebar.expander("Optimiser"):
    objectives = st.multiselect("Objectives", list(OBJECTIVES), default=["Cost", "Thickness", "Optical quality", "CO2e"])
    pop = st.select_slider("Population", [60, 100, 160, 240], 160)
    gens = st.select_slider("Generations", [30, 50, 70, 120], 70)
    seed = st.number_input("Seed", 0, 999, 1)

if not mats or not coats or len(objectives) < 2:
    st.warning("Pick at least one material, one coating and two objectives."); st.stop()

front = optimise(P, D, tuple(objectives), oq_min, margin, tcap, wcap, tuple(mats), tuple(coats), pop, gens, seed)
base = baseline_design(Spec(P, D), dict(material=b_mat, F1=b_F1, tc=b_tc, asph=int(b_asph), coat=b_coat)).iloc[0]

st.title("OptiFrugal AI")
st.caption("AI-driven design optimisation for affordable, thin, sustainable optical products - ZEISS Beyond Code 2026, Advanced Frugal Innovation")
if front.empty:
    st.error("No feasible design under these constraints. Relax the quality, thickness or safety limits, or allow more materials.")
    st.stop()
rec_i = recommend(front, weights); rec = front.iloc[rec_i]
pct = lambda a, b: 100 * (a - b) / a

tabs = st.tabs(["Recommendation", "Pareto explorer", "Feature prioritisation", "Material selection", "Model trust & method"])

# ------------------------------------------------------------------ 1. recommendation
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Unit cost", f"INR {rec.cost:.0f}", f"{-pct(base.cost, rec.cost):+.0f}% vs baseline", delta_color="inverse")
    c[1].metric("Max thickness", f"{rec.t_max:.1f} mm", f"{-pct(base.t_max, rec.t_max):+.0f}%", delta_color="inverse")
    c[2].metric("Weight", f"{rec.weight:.1f} g", f"{-pct(base.weight, rec.weight):+.0f}%", delta_color="inverse")
    c[3].metric("Optical quality", f"{rec.oq:.0f} / 100", f"{rec.oq - base.oq:+.1f} pts")
    c[4].metric("CO2e per lens", f"{rec.co2*1000:.0f} g", f"{-pct(base.co2, rec.co2):+.0f}%", delta_color="inverse")
    l, r = st.columns([3, 2])
    with l:
        st.subheader("Recommended design")
        st.write(f"**{describe(rec)}**")
        st.write(f"Impact margin **{rec.impact / IMPACT_REQ:.2f}x** pass level · edge thickness {rec.te:.1f} mm · durability {rec.dur:.0f}/100 · circularity {rec.circ:.2f}")
        diffs = []
        if int(rec.material) != int(base.material): diffs.append(f"material {MATERIALS.name[int(base.material)]} -> **{MATERIALS.name[int(rec.material)]}**")
        if int(rec.asph) != int(base.asph): diffs.append("aspheric surface " + ("added" if rec.asph else "removed"))
        if int(rec.coat) != int(base.coat): diffs.append(f"coating -> **{COATINGS.name[int(rec.coat)]}**")
        diffs.append(f"base curve {base.F1:.1f} -> {rec.F1:.1f} D, centre thickness {base.tc:.1f} -> {rec.tc:.1f} mm")
        st.markdown("**What changed vs the baseline:** " + "; ".join(diffs) + ".")
        st.markdown(f"**Why this one:** among {len(front)} Pareto-optimal designs it is closest to your priorities "
                    f"(affordability {needs['Affordability']}, thin & light {needs['Thin & light']}, clarity {needs['Optical clarity']}, eco {needs['Eco-friendliness']}). "
                    f"Quality is {'higher' if rec.oq >= base.oq else 'lower by ' + format(base.oq - rec.oq, '.1f') + ' pts'} than the baseline, and every hard constraint is met.")
    with r:
        st.subheader("Other trade-offs on the front")
        picks = {"Cheapest": front.cost.idxmin(), "Thinnest": front.t_max.idxmin(), "Lightest": front.weight.idxmin(),
                 "Best optics": front.oq.idxmax(), "Greenest": front.co2.idxmin()}
        tbl = pd.DataFrame({k: dict(Material=MATERIALS.name[int(front.material[i])], Cost=front.cost[i], Thick=front.t_max[i], Quality=front.oq[i], CO2g=front.co2[i] * 1000) for k, i in picks.items()}).T
        st.dataframe(tbl.style.format({"Cost": "{:.0f}", "Thick": "{:.1f}", "Quality": "{:.0f}", "CO2g": "{:.0f}"}), width="stretch")
    st.subheader("Baseline vs recommended")
    metrics = [("Cost (INR)", "cost"), ("Thickness (mm)", "t_max"), ("Weight (g)", "weight"), ("CO2e (g)", "co2")]
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=[m[0] for m in metrics], y=[100] * 4, marker_color=GREY, text=[f"{base[m[1]] * (1000 if m[1]=='co2' else 1):.1f}" for m in metrics])
    fig.add_bar(name="Recommended", x=[m[0] for m in metrics], y=[100 * rec[m[1]] / base[m[1]] for m in metrics], marker_color=BLUE,
                text=[f"{rec[m[1]] * (1000 if m[1]=='co2' else 1):.1f}" for m in metrics])
    fig.update_layout(barmode="group", yaxis_title="Index (baseline = 100)")
    st.plotly_chart(style(fig, 330), width="stretch")
    st.caption("Savings depend on the baseline you define in the sidebar; cost/CO2 numbers use illustrative unit economics (see 'Model trust & method').")

# ------------------------------------------------------------------ 2. pareto
with tabs[1]:
    labels = {"cost": "Cost (INR)", "t_max": "Max thickness (mm)", "weight": "Weight (g)", "oq": "Optical quality", "co2": "CO2e (kg)"}
    a, b_, c_ = st.columns(3)
    xk = a.selectbox("X axis", list(labels), 0, format_func=labels.get); yk = b_.selectbox("Y axis", list(labels), 1, format_func=labels.get)
    ck = c_.selectbox("Colour", list(labels), 3, format_func=labels.get)
    f = front.assign(Design=[describe(r) for _, r in front.iterrows()])
    fig = px.scatter(f, x=xk, y=yk, color=ck, color_continuous_scale="Blues", hover_name="Design", labels=labels)
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
    fig.add_trace(go.Scatter(x=[base[xk]], y=[base[yk]], mode="markers", name="Baseline", marker=dict(color=ORANGE, size=14, symbol="diamond", line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(x=[rec[xk]], y=[rec[yk]], mode="markers", name="Recommended", marker=dict(color=AQUA, size=16, symbol="star", line=dict(width=2, color="white"))))
    st.plotly_chart(style(fig, 460), width="stretch")
    st.caption(f"{len(front)} Pareto-optimal, constraint-satisfying designs, re-verified with the physics simulator. Orange diamond = baseline lens; green star = recommended.")
    pc = px.parallel_coordinates(front, dimensions=list(labels), color="oq", color_continuous_scale="Blues", labels=labels)
    st.plotly_chart(style(pc, 340).update_layout(margin=dict(l=70, r=40, t=50, b=10)), width="stretch")
    show = front.assign(material=[MATERIALS.name[int(i)] for i in front.material], coating=[COATINGS.name[int(i)] for i in front.coat])
    show = show[["material", "F1", "tc", "asph", "coating", "cost", "t_max", "weight", "oq", "co2", "impact", "te"]].round(2)
    st.dataframe(show, width="stretch", height=260)
    st.download_button("Download Pareto set (CSV)", show.to_csv(index=False), "optifrugal_pareto.csv")

# ------------------------------------------------------------------ 3. feature prioritisation
with tabs[2]:
    st.subheader("Which features is the customer actually paying for?")
    st.write("The optimiser only keeps a premium feature when the customer's stated needs justify its cost. Marginal effect of each feature on the recommended lens:")
    fv = feature_value(rec, None)
    fig = px.bar(fv, y="Feature", x="ExtraCostINR", orientation="h", color_discrete_sequence=[BLUE], labels={"ExtraCostINR": "Extra / (saved) cost when toggled (INR)"}, text=fv.ExtraCostINR.round(0))
    st.plotly_chart(style(fig, 280), width="stretch")
    st.dataframe(fv.rename(columns={"ExtraCostINR": "Cost change (INR)", "QualityGain": "Quality change", "ThicknessChangeMM": "Thickness change (mm)"}).round(2), width="stretch")
    st.markdown("**Need -> design lever mapping used by the recommender**")
    st.table(pd.DataFrame([
        ("Affordability", "Weight on unit cost"), ("Thin & light", "Weight on thickness and mass; favours higher index / aspheric"),
        ("Optical clarity", "Minimum optical-quality constraint + weight on quality (Abbe number, AR coating, aspheric)"),
        ("Safety & durability", "Impact margin above 3 raises the constraint; weight on durability (hydrophobic, hard materials)"),
        ("Eco-friendliness", "Weight on CO2e and recyclability (thermoplastic materials, fewer coating layers)"),
        ("Screen-use comfort", "4+ forces the blue-cut coating; otherwise it is never paid for")], columns=["Customer need", "How it steers the design"]).set_index("Customer need"))

# ------------------------------------------------------------------ 4. materials
with tabs[3]:
    st.subheader("Material selection under your constraints")
    mf = mat_front(P, D, oq_min, margin, tcap, wcap, tuple(mats), tuple(coats))
    ok = mf[mf.feasible]
    if ok.empty: st.warning("No material is feasible with these constraints.")
    else:
        fig = px.scatter(ok, x="min_cost", y="min_thickness", text="name", size=[14] * len(ok), color_discrete_sequence=[BLUE],
                         labels={"min_cost": "Lowest achievable cost (INR)", "min_thickness": "Thinnest achievable lens (mm)"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(style(fig, 420), width="stretch")
        st.dataframe(ok[["name", "min_cost", "cost_at_min_cost_t", "min_thickness", "cost_at_min_thickness", "best_weight", "best_oq", "best_co2"]].rename(columns={
            "name": "Material", "min_cost": "Min cost", "cost_at_min_cost_t": "Thickness at min cost", "min_thickness": "Min thickness",
            "cost_at_min_thickness": "Cost at min thickness", "best_weight": "Best weight (g)", "best_oq": "Best quality", "best_co2": "Min CO2e (kg)"}).round(2), width="stretch")
    bad = mf[~mf.feasible]
    if len(bad): st.caption("Infeasible under current constraints: " + ", ".join(bad.name))
    st.caption("Exhaustive grid on the ground-truth simulator (not the surrogate): for each material, the best achievable cost and thickness.")

# ------------------------------------------------------------------ 5. trust & method
with tabs[4]:
    met = json.load(open("models/metrics.json"))
    st.subheader("Surrogate accuracy")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Hold-out test** ({met['n_test']:,} unseen designs; data carries 1-2% injected noise)")
    c1.dataframe(pd.DataFrame(met["metrics"]).T.rename(columns={"r2": "R2", "mape_pct": "MAPE %"}).round(4))
    errs = {k: 100 * float(np.mean(np.abs(front["sur_" + k] - front[k]) / np.abs(front[k]))) for k in ["cost", "weight", "t_max", "oq", "co2"]}
    c2.markdown(f"**This Pareto set** ({len(front)} designs): surrogate vs ground-truth simulator")
    c2.dataframe(pd.DataFrame({"MAPE %": errs}).round(2))
    st.subheader("Pipeline")
    st.markdown("""
1. **Requirements** (power, diameter, customer needs) and constraints (impact safety, optical quality, thickness, weight).
2. **Physics-informed simulator** (sag geometry, volume integration, Abbe/Tscherning optics, drop-ball impact proxy, cost & CO2 model) generates ~39k labelled designs.
3. **Surrogate ML** (gradient-boosted regressors) predicts cost, weight, thickness, quality, impact, CO2 in microseconds.
4. **NSGA-II** searches material x curve x thickness x aspheric x coating with hard constraints; results are **re-scored by the simulator**.
5. **Recommender** converts customer needs into weights and picks the closest-to-ideal Pareto design; the dashboard explains each trade-off.
""")
    st.subheader("Assumptions and limits")
    st.markdown("""
- Material **optical and physical properties and impact strength come from published datasheets**; **cost, CO2, recyclability and the optical-quality formula are still assumptions** (full list below). The pipeline is data-agnostic: replace the simulator tables or `data/lens_design_dataset.csv` with measured or ray-traced data and retrain (`python train.py`).
- Optical quality and impact are simplified proxies; any recommendation needs ray-trace/FEA verification and physical testing before use.
- Savings are measured against the baseline you set in the sidebar.
""")

    st.subheader("Data provenance")
    try:
        st.markdown(open("data/SOURCES.md").read().replace("# Data provenance (OptiFrugal AI)", ""))
    except Exception:
        st.caption("data/SOURCES.md not found.")
