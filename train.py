"""Generate the simulated dataset and train the surrogate models."""
import time
from optifrugal.surrogate import make_dataset, train
t = time.time()
df = make_dataset()
df.to_csv("data/lens_design_dataset.csv", index=False)
print(f"dataset: {len(df)} valid designs ({time.time()-t:.1f}s)")
_, m = train(df)
for k, v in m.items():
    print(f"{k:7s} R2={v['r2']:.4f}  MAPE={v['mape_pct']:.2f}%")
print(f"done in {time.time()-t:.1f}s")
