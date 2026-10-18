"""Step 2 - descriptive stats, hypothesis tests, figures. Writes outputs/*.csv, outputs/results.json, figures/*.png"""
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, scipy.stats as st
import statsmodels.formula.api as smf
from common import *
d = load(); R = {}
ANG = ["trunk_rotation_deg", "hip_rotation_deg", "shoulder_internal_rotation_deg", "elbow_flexion_deg", "knee_flexion_deg"]

def cluster_logit(formula, data):
    return smf.logit(formula, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data["player_id"]})

# ---- 1. stroke summary
agg = d.groupby("stroke_type", observed=True).agg(
    shots=("shot_id", "size"), speed=("shot_speed_kmh", "mean"), spin=("spin_rpm", "mean"), rhs=("racket_head_speed_m_s", "mean"),
    contact_h=("contact_height_m", "mean"), success=("success", "mean"), winner=("shot_outcome", lambda x: (x == "winner").mean()),
    error=("shot_outcome", lambda x: (x == "error").mean()), net=("shot_outcome", lambda x: (x == "net").mean()))
agg.round(3).to_csv(OUT/"stroke_summary_extended.csv")
ang = d.groupby("stroke_type", observed=True)[ANG].mean().round(1); ang.to_csv(OUT/"joint_angles_by_stroke.csv")
R["stroke_success_chi2_p"] = float(st.chi2_contingency(pd.crosstab(d.stroke_type, d.success))[1])

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, (c, t) in zip(ax, [("shot_speed_kmh", "Ball speed (km/h)"), ("spin_rpm", "Spin (rpm)"), ("racket_head_speed_m_s", "Racket-head speed (m/s)")]):
    bp = a.boxplot([d.loc[d.stroke_type == s, c] for s in STROKES], tick_labels=STROKES, patch_artist=True, showfliers=False, widths=.6,
                   medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for p, s in zip(bp["boxes"], STROKES): p.set(facecolor=SC[s], alpha=.85, edgecolor="none")
    a.set_title(t); a.tick_params(axis="x", rotation=30); a.grid(axis="x", visible=False)
fig.suptitle("Stroke types differ sharply in speed, spin and racket-head speed (synthetic data)", x=.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
save(fig, "fig1_stroke_profiles")

z = (ang - d[ANG].mean()) / d[ANG].std()
fig, a = plt.subplots(figsize=(6.6, 3.2))
im = a.imshow(z.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
a.set_xticks(range(5)); a.set_xticklabels([c.replace("_deg", "").replace("_", " ") for c in ANG], rotation=20, ha="right")
a.set_yticks(range(5)); a.set_yticklabels(STROKES); a.grid(False)
for i in range(5):
    for j in range(5): a.text(j, i, f"{ang.values[i,j]:.0f}°", ha="center", va="center", fontsize=9, color=INK)
a.set_title("Mean joint angle by stroke (colour = z-score vs all shots)")
fig.colorbar(im, ax=a, shrink=.8, label="z"); save(fig, "fig2_joint_angles")

# ---- 2. serve mechanics
sv = d[d.stroke_type == "serve"].copy(); sv["serve_in"] = sv["serve_in"].astype(int)
stype = sv.groupby("serve_type").agg(n=("shot_id", "size"), speed=("serve_speed_kmh", "mean"), spin=("spin_rpm", "mean"), toss=("serve_toss_height_m", "mean"),
    shoulder_rot=("shoulder_internal_rotation_deg", "mean"), knee=("knee_flexion_deg", "mean"), serve_in=("serve_in", "mean"), success=("success", "mean"))
stype.round(3).to_csv(OUT/"serve_types.csv")
R["serve_type_speed_anova_p"] = float(st.f_oneway(*[g.serve_speed_kmh for _, g in sv.groupby("serve_type")])[1])
R["serve_type_spin_anova_p"] = float(st.f_oneway(*[g.spin_rpm for _, g in sv.groupby("serve_type")])[1])
R["serve_type_in_chi2_p"] = float(st.chi2_contingency(pd.crosstab(sv.serve_type, sv.serve_in))[1])
for c in ["serve_toss_height_m", "shoulder_internal_rotation_deg", "knee_flexion_deg", "trunk_rotation_deg", "racket_head_speed_m_s"]:
    r, p = st.pearsonr(sv[c], sv.serve_speed_kmh); R[f"serve_speed_vs_{c}_r"] = float(r)
m = cluster_logit("serve_in ~ serve_speed_kmh + C(serve_type)", sv)
R["serve_in_per10kmh_OR"] = float(np.exp(m.params["serve_speed_kmh"] * 10)); R["serve_in_speed_p"] = float(m.pvalues["serve_speed_kmh"])
R["serve_in_overall"] = float(sv.serve_in.mean()); R["serve_success_by_in"] = sv.groupby("serve_in").success.mean().round(3).to_dict()

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
for t, col in zip(["flat", "slice", "kick"], PAL[:3]):
    ax[0].hist(sv.loc[sv.serve_type == t, "serve_speed_kmh"], bins=40, histtype="step", lw=1.8, color=col, label=t, density=True)
ax[0].set_title("Serve speed by serve type"); ax[0].set_xlabel("km/h"); ax[0].legend()
q = pd.qcut(sv.serve_speed_kmh, 10); g = sv.groupby(q, observed=True).agg(x=("serve_speed_kmh", "mean"), y=("serve_in", "mean"), n=("serve_in", "size"))
se_ = np.sqrt(g.y * (1 - g.y) / g.n)
ax[1].errorbar(g.x, g.y, 1.96 * se_, fmt="o-", color=PAL[0], lw=2, ms=6, mfc=SURF, mec=PAL[0], ecolor=MUTED, capsize=2)
ax[1].set_title("Serve-in rate by speed decile"); ax[1].set_xlabel("serve speed (km/h)"); ax[1].set_ylabel("serve-in rate"); ax[1].set_ylim(.4, .7)
ax[2].scatter(sv.serve_toss_height_m, sv.serve_speed_kmh, s=3, alpha=.25, color=PAL[0], linewidths=0)
ax[2].set_title("Toss height vs serve speed"); ax[2].set_xlabel("toss height (m)"); ax[2].set_ylabel("km/h")
save(fig, "fig3_serve")

# ---- 3. footwork / movement
fw = d.groupby("skill_level", observed=True).agg(split_step_latency=("split_step_latency_s", "mean"), split_step_used=("split_step_used", "mean"),
    lateral=("lateral_displacement_m", "mean"), distance=("movement_distance_m", "mean"), accel=("peak_acceleration_m_s2", "mean"),
    decel=("peak_deceleration_m_s2", "mean"), success=("success", "mean"))
fw.round(3).to_csv(OUT/"footwork_by_skill.csv")
pl = d.groupby(["player_id", "skill_level"], observed=True).split_step_latency_s.mean().reset_index()   # player-level unit of analysis
R["split_latency_kruskal_p_player_level"] = float(st.kruskal(*[g.split_step_latency_s for _, g in pl.groupby("skill_level", observed=True)])[1])
R["split_latency_rho_vs_skill_rank"] = float(st.spearmanr(pl.skill_level.cat.codes, pl.split_step_latency_s)[0])
for c in ["split_step_latency_s", "peak_acceleration_m_s2", "peak_deceleration_m_s2", "lateral_displacement_m", "movement_distance_m", "change_direction_angle_deg"]:
    m = cluster_logit(f"success ~ {c} + C(stroke_type) + C(skill_level)", d)
    sd = d[c].std(); R[f"logit_{c}_OR_per_SD"] = float(np.exp(m.params[c] * sd)); R[f"logit_{c}_p"] = float(m.pvalues[c])
ss = d.groupby("split_step_used").success.mean(); R["success_split_step_used"] = ss.round(3).to_dict()
mv = d.groupby("rally_context").agg(distance=("movement_distance_m", "mean"), lateral=("lateral_displacement_m", "mean"), accel=("peak_acceleration_m_s2", "mean"),
                                   decel=("peak_deceleration_m_s2", "mean"), success=("success", "mean")); mv.round(3).to_csv(OUT/"movement_by_rally_context.csv")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
bp = ax[0].boxplot([pl.loc[pl.skill_level == s, "split_step_latency_s"] for s in SKILLS], tick_labels=SKILLS, patch_artist=True, showfliers=False, widths=.6, medianprops=dict(color=INK, lw=1.4))
for p, s in zip(bp["boxes"], SKILLS): p.set(facecolor=KC[s], alpha=.85, edgecolor="none")
ax[0].set_title("Split-step latency by skill"); ax[0].set_ylabel("seconds"); ax[0].grid(axis="x", visible=False); ax[0].tick_params(axis="x", rotation=20)
for i, c in enumerate(["peak_acceleration_m_s2", "peak_deceleration_m_s2"]):
    q = pd.qcut(d[c], 10); g = d.groupby(q, observed=True).agg(x=(c, "mean"), y=("success", "mean"), n=("success", "size")); se_ = np.sqrt(g.y * (1 - g.y) / g.n)
    ax[1 + i].errorbar(g.x, g.y, 1.96 * se_, fmt="o-", color=PAL[i], lw=2, ms=6, mfc=SURF, mec=PAL[i], ecolor=MUTED, capsize=2)
    ax[1 + i].set_title(f"Success by {'acceleration' if i == 0 else 'deceleration'} decile"); ax[1 + i].set_xlabel("m/s²"); ax[1 + i].set_ylim(.5, .7)
ax[1].set_ylabel("success rate"); save(fig, "fig4_footwork")

# ---- 4. kinetic chain
rows = []
for s in STROKES:
    g = d[d.stroke_type == s]
    r = {"stroke": s, "rhs_vs_ball_speed_r": g.racket_head_speed_m_s.corr(g.shot_speed_kmh)}
    for c in ANG: r[c + "_vs_rhs_r"] = g[c].corr(g.racket_head_speed_m_s)
    rows.append(r)
kc = pd.DataFrame(rows).set_index("stroke"); kc.round(3).to_csv(OUT/"kinetic_chain_correlations.csv")
m = smf.ols("shot_speed_kmh ~ racket_head_speed_m_s + C(stroke_type)", d).fit(cov_type="cluster", cov_kwds={"groups": d.player_id})
R["ball_speed_per_1ms_rhs_kmh"] = float(m.params["racket_head_speed_m_s"]); R["ball_speed_rhs_model_r2"] = float(m.rsquared)
m2 = smf.ols("racket_head_speed_m_s ~ " + " + ".join(ANG) + " + C(stroke_type)", d).fit(cov_type="cluster", cov_kwds={"groups": d.player_id})
R["rhs_from_angles_r2"] = float(m2.rsquared); R["rhs_angle_coefs"] = {k: float(v) for k, v in m2.params.items() if k in ANG}
m3 = smf.ols("racket_head_speed_m_s ~ " + " + ".join(ANG) + " + C(stroke_type) + latent_power_index + C(skill_level)", d).fit()
R["rhs_from_angles_plus_athlete_r2"] = float(m3.rsquared)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
for s in STROKES:
    g = d[d.stroke_type == s].sample(1200, random_state=0)
    ax[0].scatter(g.racket_head_speed_m_s, g.shot_speed_kmh, s=4, alpha=.35, color=SC[s], linewidths=0, label=s)
ax[0].set_xlabel("racket-head speed (m/s)"); ax[0].set_ylabel("ball speed (km/h)"); ax[0].set_title("Racket-head speed vs ball speed")
ax[0].legend(markerscale=4, ncol=2)
cc = kc[[c + "_vs_rhs_r" for c in ANG]]; im = ax[1].imshow(cc.values, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([c.replace("_deg", "").replace("_", " ") for c in ANG], rotation=25, ha="right"); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(STROKES); ax[1].grid(False)
for i in range(5):
    for j in range(5): ax[1].text(j, i, f"{cc.values[i,j]:.2f}", ha="center", va="center", fontsize=8.5)
ax[1].set_title("Joint angle vs racket-head speed (r)"); fig.colorbar(im, ax=ax[1], shrink=.8); save(fig, "fig5_kinetic_chain")

# ---- 5. fatigue & workload
fb = d.groupby("shot_number_in_session").agg(fatigue=("fatigue_index_0_1", "mean"), hr=("heart_rate_bpm", "mean"), rec=("recovery_time_s", "mean"),
    success=("success", "mean"), speed=("shot_speed_kmh", "mean"), lat=("split_step_latency_s", "mean"), accel=("peak_acceleration_m_s2", "mean"))
fb.round(4).to_csv(OUT/"fatigue_by_shot_number.csv")
# within-player fatigue effects: player fixed effects, cluster-robust SEs
for y in ["shot_speed_kmh", "split_step_latency_s", "peak_acceleration_m_s2", "recovery_time_s", "heart_rate_bpm", "racket_head_speed_m_s"]:
    m = smf.ols(f"{y} ~ fatigue_index_0_1 + C(stroke_type) + C(player_id)", d).fit(cov_type="cluster", cov_kwds={"groups": d.player_id})
    R[f"fe_{y}_per_0.1_fatigue"] = float(m.params["fatigue_index_0_1"] * .1); R[f"fe_{y}_p"] = float(m.pvalues["fatigue_index_0_1"])
m = cluster_logit("success ~ fatigue_index_0_1 + C(stroke_type) + C(skill_level)", d)
R["success_logit_fatigue_OR_per_0.1"] = float(np.exp(m.params["fatigue_index_0_1"] * .1)); R["success_logit_fatigue_p"] = float(m.pvalues["fatigue_index_0_1"])
d["fat_bin"] = pd.cut(d.fatigue_index_0_1, [-.001, .2, .4, .6, .8, 1.0]); fs = d.groupby("fat_bin", observed=True).success.agg(["mean", "size"]); fs.to_csv(OUT/"success_by_fatigue_bin.csv")
R["fatigue_by_skill"] = d.groupby("skill_level", observed=True).fatigue_index_0_1.mean().round(3).to_dict()
sess = d.groupby("session_id").agg(load=("impact_load_proxy_au", "sum"), shoulder=("shoulder_load_proxy_au", "sum"), dist=("movement_distance_m", "sum"), rpe=("session_rpe_0_10", "first"),
    dur=("session_duration_min", "first"), temp=("environment_temp_c", "first"), sleep=("sleep_hours_pre_session", "first"), maxfat=("fatigue_index_0_1", "max"), hr=("heart_rate_bpm", "mean"))
R["session_rpe_vs_load_r"] = float(sess.rpe.corr(sess.load)); R["session_rpe_vs_duration_r"] = float(sess.rpe.corr(sess.dur)); R["session_rpe_vs_dist_r"] = float(sess.rpe.corr(sess.dist))
R["session_maxfat_vs_sleep_r"] = float(sess.maxfat.corr(sess.sleep)); R["session_maxfat_vs_temp_r"] = float(sess.maxfat.corr(sess.temp))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
ax[0].plot(fb.index, fb.fatigue, color=PAL[1], lw=2); ax[0].set_title("Fatigue index climbs through a session"); ax[0].set_xlabel("shot number"); ax[0].set_ylabel("mean fatigue (0–1)")
ax[1].plot(fb.index, fb.hr, color=PAL[3], lw=2); ax[1].set_title("Heart rate"); ax[1].set_xlabel("shot number"); ax[1].set_ylabel("bpm")
fs2 = d.groupby("fat_bin", observed=True).success.agg(["mean", "size"]); se_ = np.sqrt(fs2["mean"] * (1 - fs2["mean"]) / fs2["size"])
ax[2].bar(range(5), fs2["mean"], yerr=1.96 * se_, color=PAL[0], width=.6, error_kw=dict(ecolor=MUTED, capsize=3)); ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"])
ax[2].set_ylim(.5, .68); ax[2].set_title("Shot success by fatigue band"); ax[2].set_xlabel("fatigue index"); ax[2].set_ylabel("success rate"); ax[2].grid(axis="x", visible=False)
save(fig, "fig6_fatigue")

# ---- 6. what separates shot success? skill, and players
sk = []
for s in SKILLS:
    m_, lo, hi = boot_ci_by_player(d[d.skill_level == s], "success"); sk.append(dict(skill=s, success=m_, lo=lo, hi=hi, players=d[d.skill_level == s].player_id.nunique()))
sk = pd.DataFrame(sk); sk.round(3).to_csv(OUT/"success_by_skill.csv", index=False)
pp = d.groupby("player_id").success.mean(); binom_sd = np.sqrt(.6 * .4 / 250)
R["player_success_sd"] = float(pp.std()); R["player_success_sd_if_pure_chance"] = float(binom_sd)
ps = d.groupby(["player_id", "skill_level"], observed=True).agg(success=("success", "mean"), speed=("shot_speed_kmh", "mean")).reset_index()
R["player_level_success_vs_latent_power_r"] = float(pd.Series(pp.values).corr(d.groupby("player_id").latent_power_index.first().reset_index(drop=True)))
R["player_level_success_vs_latent_mobility_r"] = float(pd.Series(pp.values).corr(d.groupby("player_id").latent_mobility_index.first().reset_index(drop=True)))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].errorbar(range(4), sk.success, yerr=[sk.success - sk.lo, sk.hi - sk.success], fmt="o", color=PAL[0], ms=8, mfc=SURF, mec=PAL[0], mew=2, ecolor=MUTED, capsize=4)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(SKILLS); ax[0].set_ylabel("shot success rate"); ax[0].set_title("Success rises with skill (95% CI, resampling players)"); ax[0].grid(axis="x", visible=False)
for s in SKILLS: ax[1].scatter(ps.loc[ps.skill_level == s, "speed"], ps.loc[ps.skill_level == s, "success"], s=16, color=KC[s], label=s, alpha=.85, linewidths=0)
ax[1].set_xlabel("player mean ball speed (km/h)"); ax[1].set_ylabel("player success rate"); ax[1].set_title("One dot per player"); ax[1].legend()
save(fig, "fig7_skill")

json.dump(R, open(OUT/"results.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
for f in ["stroke_summary_extended", "serve_types", "footwork_by_skill", "kinetic_chain_correlations", "success_by_skill", "success_by_fatigue_bin"]:
    print("\n", f); print(pd.read_csv(OUT/f"{f}.csv").to_string())
