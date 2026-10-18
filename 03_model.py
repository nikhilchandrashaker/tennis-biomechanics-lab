"""Step 3 - predict shot_success with player-separated evaluation. Writes outputs/model_*.csv, figures/fig8-9"""
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from common import *

d = load(); SEED = 42
# Leakage guards: shot_outcome (defines the label), serve_in/serve_speed (serve-only / redundant),
# latent_* (generator internals that would not exist with real data), ids, dates.
SHOT_NUM = ["shot_speed_kmh", "spin_rpm", "racket_head_speed_m_s", "trunk_rotation_deg", "shoulder_internal_rotation_deg", "elbow_flexion_deg", "knee_flexion_deg",
            "hip_rotation_deg", "contact_height_m", "split_step_latency_s", "lateral_displacement_m", "movement_distance_m", "peak_acceleration_m_s2",
            "peak_deceleration_m_s2", "change_direction_angle_deg", "impact_load_proxy_au", "estimated_ground_reaction_force_bw", "shoulder_load_proxy_au",
            "fatigue_index_0_1", "heart_rate_bpm", "recovery_time_s", "shot_number_in_session"]
SHOT_CAT = ["stroke_type", "rally_context"]
CTX_NUM = ["age_years", "height_cm", "body_mass_kg", "training_years", "environment_temp_c", "session_duration_min", "sleep_hours_pre_session"]
CTX_CAT = ["skill_level", "surface", "handedness"]
d["split_step_used"] = d["split_step_used"].astype(int); SHOT_NUM.append("split_step_used")
for c in SHOT_CAT + CTX_CAT: d[c] = d[c].astype(str)
y, g = d.success.values, d.player_id.values

def pre(num, cat): return ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
def models(num, cat):
    return {"Logistic regression": Pipeline([("p", pre(num, cat)), ("m", LogisticRegression(C=1.0, max_iter=2000))]),
            "Random forest": Pipeline([("p", pre(num, cat)), ("m", RandomForestClassifier(300, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=SEED))]),
            "Gradient boosting": Pipeline([("p", pre(num, cat)), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=200, l2_regularization=5, random_state=SEED))])}
FEATS = {"A. Shot mechanics + movement + fatigue": (SHOT_NUM, SHOT_CAT), "B. A + athlete/session context": (SHOT_NUM + CTX_NUM, SHOT_CAT + CTX_CAT),
         "C. Athlete skill only": ([], ["skill_level"]), "D. Stroke type only": ([], ["stroke_type"])}

def metrics(yt, p): return dict(auc=roc_auc_score(yt, p), logloss=log_loss(yt, p), brier=brier_score_loss(yt, p), acc=accuracy_score(yt, p > .5))

# ---- 5-fold GroupKFold: every player is entirely in train OR test within a fold
res, oof = [], {}
gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for fname, (num, cat) in FEATS.items():
    mods = {"Logistic regression": models(num, cat)["Logistic regression"]} if fname[0] in "CD" else models(num, cat)
    for mname, mod in mods.items():
        p = np.zeros(len(d)); fold = []
        for tr, te in gkf.split(d, y, g):
            assert not set(g[tr]) & set(g[te])
            mod.fit(d.iloc[tr][num + cat], y[tr]); p[te] = mod.predict_proba(d.iloc[te][num + cat])[:, 1]; fold.append(roc_auc_score(y[te], p[te]))
        m = metrics(y, p); m.update(features=fname, model=mname, auc_fold_sd=np.std(fold)); res.append(m); oof[(fname, mname)] = p
base = metrics(y, np.full(len(y), y.mean())); res.append(dict(base, features="Baseline (always predict base rate)", model="Constant", auc_fold_sd=0))
R = pd.DataFrame(res)[["features", "model", "auc", "auc_fold_sd", "logloss", "brier", "acc"]]; R.round(4).to_csv(OUT/"model_cv_results.csv", index=False); print(R.round(4).to_string())

# ---- Final hold-out: 40 players never seen in training or tuning
tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(d, y, g))
assert not set(g[tr]) & set(g[te]); print("train players", len(set(g[tr])), "test players", len(set(g[te])))
num, cat = FEATS["B. A + athlete/session context"]; Xtr, Xte = d.iloc[tr][num + cat], d.iloc[te][num + cat]
H = []
for mname, mod in models(num, cat).items():
    mod.fit(Xtr, y[tr]); p = mod.predict_proba(Xte)[:, 1]; H.append(dict(model=mname, **metrics(y[te], p)))
H.append(dict(model="Base rate", **metrics(y[te], np.full(len(te), y[tr].mean()))))
H = pd.DataFrame(H); H.round(4).to_csv(OUT/"model_holdout_results.csv", index=False); print(H.round(4).to_string())

# ---- Inference on the hold-out players: permutation importance (AUC drop) for the RF + calibration
rf = models(num, cat)["Random forest"]; rf.fit(Xtr, y[tr])
pi = permutation_importance(rf, Xte, y[te], scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": num + cat, "auc_drop": pi.importances_mean, "sd": pi.importances_std}).sort_values("auc_drop", ascending=False); imp.round(4).to_csv(OUT/"model_permutation_importance.csv", index=False)
lr = models(num, cat)["Logistic regression"]; lr.fit(Xtr, y[tr]); names = lr.named_steps["p"].get_feature_names_out()
co = pd.DataFrame({"feature": names, "std_coef": lr.named_steps["m"].coef_[0]}); co["odds_ratio_per_SD"] = np.exp(co.std_coef); co = co.reindex(co.std_coef.abs().sort_values(ascending=False).index)
co.round(4).to_csv(OUT/"model_logit_coefficients.csv", index=False); print(imp.head(10).round(4).to_string()); print(co.head(12).round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.2, 1]))
top = imp.head(10).iloc[::-1]; ax[0].barh(top.feature.str.replace("_", " "), top.auc_drop, xerr=top.sd, color=PAL[0], height=.6, error_kw=dict(ecolor=MUTED, capsize=2))
ax[0].set_xlabel("drop in AUC when shuffled (hold-out players)"); ax[0].set_title("What the random forest actually uses"); ax[0].grid(axis="y", visible=False)
for (fn, mn), c, ls in [(("B. A + athlete/session context", "Logistic regression"), PAL[0], "-"), (("A. Shot mechanics + movement + fatigue", "Logistic regression"), PAL[1], "-")]:
    fr, mp = calibration_curve(y, oof[(fn, mn)], n_bins=8, strategy="quantile"); ax[1].plot(mp, fr, "o-", color=c, lw=2, ms=5, label=fn.split(".")[0] + ". " + fn.split(". ")[1])
ax[1].plot([.5, .72], [.5, .72], color=MUTED, ls="--", lw=1); ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed success rate"); ax[1].set_title("Calibration (out-of-fold, unseen players)"); ax[1].legend(fontsize=8)
save(fig, "fig8_model")
