# Synthetic Tennis Biomechanics Dataset

**Important: every record in this package is synthetic. No rows are real athlete observations.**
The measurements are simulated with plausible ranges and simple relationships for demonstration, education, prototyping, and testing analysis workflows. They must not be used as empirical evidence, injury-risk prediction, clinical decisions, or claims about real tennis players.

## Contents
- `shots.csv`: 50,000 synthetic shot-level records covering serves, forehands, backhands, volleys, and overheads.
- `athletes.csv`: 200 synthetic participant profiles.
- `sessions.csv`: 1,000 synthetic sessions.
- `data_dictionary.csv`: column definitions and units.
- `stroke_summary.csv`: descriptive summary by stroke type.
- `synthetic_tennis_biomechanics_dataset.xlsx`: workbook with all tables in separate sheets.

## Data design
- 200 synthetic players × 5 sessions per player × 50 shots per session = 50,000 shot records.
- Shot-level fields cover ball speed/spin, racket-head speed, selected joint angles, contact height, footwork/movement, load proxies, fatigue, simulated heart rate, recovery proxy, and outcome labels.
- Serve-only fields are blank for non-serve shots.
- Variables ending in `_synthetic` are explicitly simulated. Load and ground-reaction-force fields are modelled proxies, not force-plate measurements.
- IDs and dates are fictional identifiers/timeline values.

## Suggested uses
- Practice data cleaning, visualization, feature engineering, model pipeline development, and dashboard prototyping.
- Test relational joins using `player_id` and `session_id`.
- Do not present the generated distributions as real-world population estimates.
