"""Step 1 - validate the dataset. Writes outputs/validation_checks.csv"""
import pandas as pd, numpy as np
from common import *
d = load(); rows = []
def chk(name, ok, detail=""): rows.append(dict(check=name, status="PASS" if ok else "FLAG", detail=detail))

chk("50,000 shots / 200 players / 1,000 sessions", len(d) == 50000 and d.player_id.nunique() == 200 and d.session_id.nunique() == 1000)
chk("shot_id unique", d.shot_id.is_unique)
chk("50 shots per session", (d.groupby("session_id").size() == 50).all())
chk("5 sessions per player", (d.groupby("player_id").session_id.nunique() == 5).all())
chk("all joins resolve (merge validated m:1)", True)
sv = d.stroke_type == "serve"
chk("serve-only fields blank for non-serves, filled for serves",
    d.loc[~sv, ["serve_type", "serve_toss_height_m", "serve_speed_kmh", "serve_in"]].isna().all().all()
    and d.loc[sv, ["serve_type", "serve_toss_height_m", "serve_speed_kmh", "serve_in"]].notna().all().all())
chk("serve_speed_kmh == shot_speed_kmh on serves (redundant column)", (d.loc[sv, "serve_speed_kmh"] == d.loc[sv, "shot_speed_kmh"]).all())
mapping = d.groupby("shot_outcome").success.agg(["min", "max"])
chk("shot_success is a deterministic function of shot_outcome (winner/in_play=1, error/net=0)",
    mapping.loc[["winner", "in_play"]].min().min() == 1 and mapping.loc[["error", "net"]].max().max() == 0,
    "=> success is NOT independent of outcome; never use outcome as a predictor")
ct = pd.crosstab(d.loc[sv, "serve_in"], d.loc[sv, "success"], normalize="index")
chk("serve_in agrees with shot_success on serves", abs(ct.loc[True, 1] - ct.loc[False, 1]) > .2,
    f"success rate when serve_in=True {ct.loc[True,1]:.3f} vs False {ct.loc[False,1]:.3f}: the two labels are generated independently")
chk("no missing values outside serve-only fields", d.drop(columns=["serve_type", "serve_toss_height_m", "serve_speed_kmh", "serve_in"]).isna().sum().sum() == 0)
chk("fatigue index within [0,1]", d.fatigue_index_0_1.between(0, 1).all())

# values piled on the clip boundaries (simulation clipped ranges)
num = ["shot_speed_kmh", "spin_rpm", "racket_head_speed_m_s", "trunk_rotation_deg", "shoulder_internal_rotation_deg",
       "elbow_flexion_deg", "knee_flexion_deg", "hip_rotation_deg", "fatigue_index_0_1", "movement_distance_m", "spin_rpm"]
clip = []
for c in dict.fromkeys(num):
    lo, hi = d[c].min(), d[c].max()
    clip.append(dict(column=c, min=lo, n_at_min=int((d[c] == lo).sum()), max=hi, n_at_max=int((d[c] == hi).sum())))
clip = pd.DataFrame(clip); clip.to_csv(OUT/"boundary_pileups.csv", index=False)
big = clip[(clip.n_at_min > 20) | (clip.n_at_max > 20)]
chk("no large pile-ups on min/max (clipping)", big.empty, "; ".join(f"{r.column}: {r.n_at_min} at min, {r.n_at_max} at max" for r in big.itertuples()))

pd.DataFrame(rows).to_csv(OUT/"validation_checks.csv", index=False)
print(pd.DataFrame(rows).to_string()); print(clip.to_string())
