# OptiFrugal AI  

Design-optimisation platform that finds lens designs that are **cheaper, thinner/lighter, greener and at least as good optically**,
and ties each design to what the customer actually needs.

## Run
```bash
pip install -r requirements.txt
python train.py          # builds data/lens_design_dataset.csv + models/ (about 10 s)
streamlit run app.py     # dashboard
python benchmark.py      # NSGA-II vs random search + savings table -> data/benchmark.csv
```

## How it maps to the ZEISS AFI brief
| Brief focus area | Where in OptiFrugal |
|---|---|
| Design optimisation | NSGA-II over material, base curve, thickness, aspheric, coating; simulator re-verification |
| Material selection | "Material selection" tab: best achievable cost / thickness per material under constraints |
| Feature prioritisation | Customer needs -> objective weights / constraints; "Feature prioritisation" tab shows what each premium feature costs |
| Sustainability | CO2e objective, circularity (recyclability) score |
| "Smaller, cheaper, better" | Thickness, weight, cost AND optical quality optimised together |

## Architecture
`optifrugal/simulator.py` physics-informed ground truth (replace with real data) ->
`surrogate.py` gradient-boosted surrogates -> `optimizer.py` NSGA-II + constraints + recommender -> `app.py` dashboard.

## Honest limitations
- Material optical/physical properties and impact strength come from published datasheets (see `data/SOURCES.md`); cost, CO2, recyclability and the optical-quality formula are still assumptions, not ZEISS data.
- Optical quality and impact safety are simplified proxies; real use needs ray-tracing / FEA and physical testing.
- Savings are measured against an editable baseline lens; they are not a claim about real ZEISS products.
- On easy (loosely constrained) cases NSGA-II is only marginally better than random search; it is clearly better on tightly constrained cases (e.g. -9 D: hypervolume 0.78 vs 0.62, with random search finding only 181 feasible designs).

## Roadmap
- Source the remaining assumptions (processing cost, CO2, recyclability) from public data
- Design-flaw finder for an existing lens design, plus sensitivity analysis
- Hook for a CAD / ray-tracing API so the simulator can be swapped for real tools
