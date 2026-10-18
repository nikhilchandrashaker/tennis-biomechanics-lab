"""Shared loading + plotting helpers."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, OUT = ROOT/"data", ROOT/"figures", ROOT/"outputs"
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# categorical palette (fixed order) on a light surface
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, SUB, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
STROKES = ["serve", "forehand", "backhand", "volley", "overhead"]
SKILLS = ["recreational", "club", "competitive", "elite"]
SC = dict(zip(STROKES, PAL[:5])); KC = dict(zip(SKILLS, PAL[:4]))

def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": SUB, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 160,
        "savefig.bbox": "tight"})
style()

def load():
    s = pd.read_csv(DATA/"shots.csv"); se = pd.read_csv(DATA/"sessions.csv"); a = pd.read_csv(DATA/"athletes.csv")
    d = s.merge(se, on=["session_id", "player_id"], validate="m:1").merge(a, on="player_id", validate="m:1")
    d["success"] = d["shot_success_synthetic"].astype(int)
    d["stroke_type"] = pd.Categorical(d["stroke_type"], STROKES)
    d["skill_level"] = pd.Categorical(d["skill_level"], SKILLS, ordered=True)
    d = d.rename(columns={c: c.replace("_synthetic", "") for c in d.columns})
    d = d.rename(columns={"serve_toss_height_m_s": "serve_toss_height_m"})  # source typo
    return d

def boot_ci_by_player(d, col, groupcol=None, n=500, seed=0):
    """Mean of `col` with a percentile CI from resampling PLAYERS (not shots)."""
    rng = np.random.default_rng(seed)
    pm = d.groupby("player_id")[col].agg(["sum", "count"])
    idx = np.arange(len(pm)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx)); out.append(pm["sum"].values[b].sum() / pm["count"].values[b].sum())
    return d[col].mean(), *np.percentile(out, [2.5, 97.5])

def save(fig, name):
    fig.savefig(FIG/f"{name}.png"); plt.close(fig)
