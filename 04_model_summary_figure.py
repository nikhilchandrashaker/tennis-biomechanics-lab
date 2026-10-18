"""Step 4 - summary figure of CV results (reads outputs/model_cv_results.csv)."""
import pandas as pd
from common import *
R = pd.read_csv(OUT/"model_cv_results.csv"); b = R[R.model != "Constant"].copy()
b["label"] = b.features.str[:1] + " · " + b.model; b = b.iloc[::-1]
col = {"Logistic regression": PAL[0], "Random forest": PAL[1], "Gradient boosting": PAL[2]}
fig, a = plt.subplots(figsize=(7.5, 3.6))
a.barh(b.label, b.auc - .5, left=.5, color=[col[m] for m in b.model], height=.65)
a.axvline(.5, color=INK, lw=1); a.set_xlim(.47, .58); a.set_xlabel("AUC, 5-fold CV with players held out (0.5 = coin flip)")
a.set_title("Predictive skill is modest"); a.grid(axis="y", visible=False)
for i, v in enumerate(b.auc): a.text(max(v, .5) + .002, i, f"{v:.3f}", va="center", fontsize=8.5, color=SUB)
save(fig, "fig9_model_comparison")
