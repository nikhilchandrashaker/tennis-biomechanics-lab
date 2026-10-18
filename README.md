# Tennis performance analysis (synthetic biomechanics data)

Question: how do stroke mechanics, movement and fatigue relate to shot outcomes?

**All data are synthetic.** Results show what the simulation encodes and demonstrate the method; they are not findings about real players.

Run: `pip install -r requirements.txt && bash src/run_all.sh` (about 8 min on 2 cores; seeds fixed).

- `src/01_validate.py` - data checks -> `outputs/validation_checks.csv`, `outputs/boundary_pileups.csv`
- `src/02_analysis.py` - tables, tests, figures 1-7 -> `outputs/`, `figures/`, `outputs/results.json`
- `src/03_model.py` - player-grouped CV + 40-player hold-out -> `outputs/model_*.csv`, figure 8
- `src/04_model_summary_figure.py` - figure 9

Key leakage guards: `shot_outcome`, serve-only columns, `latent_*` indices, IDs and dates are never model inputs; every split is by player.
