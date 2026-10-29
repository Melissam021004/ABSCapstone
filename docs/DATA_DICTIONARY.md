# Data Dictionary — `challenge_inventory_2026`

Generated 2026-10-08 by `scripts/09_data_dictionary.py`. Do not edit by hand — edit the REGISTRY in that script and re-run.

**720,209 rows × 166 columns.** One row per pitch, keyed on `game_pk` + `at_bat_number` + `pitch_number`.

## Modelling roles

| Role | Columns | Meaning |
|---|---:|---|
| target | 3 | What you predict. Never an input. |
| **LEAKAGE RISK** | 5 | Known only after the challenge resolved. Using these as features inflates performance and invalidates the model. |
| feature | 50 | Safe model input — known at the moment of the decision. |
| identifier | 5 | Keys and names. For joining and inspection. |
| metadata | 16 | Provenance, flags and bookkeeping. |
| UNDOCUMENTED | 87 | Present in the table but not yet defined here. |

**Targets:** `is_overturned`, `was_challenged`, `would_be_overturned`

> **Before Week 7, drop these from any feature matrix:**  
> `challenger_role`, `challenger_id`, `challenging_side`, `challenge_call_code`, `challenge_description`
>
> Each is known only once a challenge has happened and resolved. This is the leak your Week 8 notes warn about.

## Data quality

- **8 columns are entirely null:** `spin_dir`, `spin_rate_deprecated`, `break_angle_deprecated`, `break_length_deprecated`, `tfs_deprecated`, `tfs_zulu_deprecated`, `umpire`, `sv_id`. Decide in Week 6 whether to drop them.

- **44 columns are under 95% populated.** Each needs a Week 6 decision: drop the row, impute, or add a missing-indicator.

| Column | Non-null % | Source |
|---|---:|---|
| `challenger_role` | 1.1% | MLB Stats API (script 02) |
| `challenger_id` | 1.1% | MLB Stats API (script 02) |
| `challenger_name` | 1.1% | MLB Stats API (script 02) |
| `challenging_side` | 1.1% | MLB Stats API (script 02) |
| `challenge_call_code` | 1.1% | MLB Stats API (script 02) |
| `challenge_description` | 1.1% | MLB Stats API (script 02) |
| `is_overturned` | 1.1% | MLB Stats API (script 02) |
| `on_3b` | 9.7% | Statcast (script 01) |
| `miss_distance` | 10.7% | Statcast (script 01) |
| `estimated_ba_using_speedangle` | 17.0% | Statcast (script 01) |
| `estimated_slg_using_speedangle` | 17.0% | Statcast (script 01) |
| `launch_speed_angle` | 17.3% | Statcast (script 01) |
| `hc_x` | 17.4% | Statcast (script 01) |
| `hc_y` | 17.4% | Statcast (script 01) |
| `bb_type` | 17.4% | Statcast (script 01) |
| `on_2b` | 19.2% | Statcast (script 01) |
| `hit_location` | 22.3% | Statcast (script 01) |
| `estimated_woba_using_speedangle` | 25.3% | Statcast (script 01) |
| `woba_denom` | 25.4% | Statcast (script 01) |
| `des` | 25.6% | Statcast (script 01) |
| `events` | 25.7% | Statcast (script 01) |
| `babip_value` | 25.7% | Statcast (script 01) |
| `woba_value` | 25.7% | Statcast (script 01) |
| `iso_value` | 25.7% | Statcast (script 01) |
| `on_1b` | 31.2% | Statcast (script 01) |
| `launch_speed` | 33.1% | Statcast (script 01) |
| `hit_distance_sc` | 33.1% | Statcast (script 01) |
| `launch_angle` | 33.1% | Statcast (script 01) |
| `hyper_speed` | 33.3% | Statcast (script 01) |
| `intercept_ball_minus_batter_pos_y_inches` | 46.1% | Statcast (script 01) |
| `intercept_ball_minus_batter_pos_x_inches` | 46.1% | Statcast (script 01) |
| `attack_direction` | 46.1% | Statcast (script 01) |
| `swing_length` | 46.1% | Statcast (script 01) |
| `attack_angle` | 46.1% | Statcast (script 01) |
| `bat_speed` | 46.1% | Statcast (script 01) |
| `swing_path_tilt` | 46.1% | Statcast (script 01) |
| `catcher_rv_12` | 81.6% | Savant catcher framing (script 04) |
| `catcher_rv_13` | 81.6% | Savant catcher framing (script 04) |
| `catcher_rv_11` | 81.6% | Savant catcher framing (script 04) |
| `catcher_rv_14` | 81.6% | Savant catcher framing (script 04) |
| `catcher_rv_tot` | 81.6% | Savant catcher framing (script 04) |
| `catcher_rv_16` | 81.6% | Savant catcher framing (script 04) |
| `pitcher_days_until_next_game` | 92.7% | Statcast (script 01) |
| `pitcher_days_since_prev_game` | 94.3% | Statcast (script 01) |

- **87 columns are undocumented** (Statcast columns this project doesn't use): `player_name`, `events`, `spin_dir`, `spin_rate_deprecated`, `break_angle_deprecated`, `break_length_deprecated`, `des`, `game_type`, `home_team`, `away_team`, `hit_location`, `bb_type`, `game_year`, `hc_x`, `hc_y`, `tfs_deprecated`, `tfs_zulu_deprecated`, `umpire`, `sv_id`, `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az`...

## Statcast (script 01)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `away_score` | Int64 | feature | 100% | 0.0 … 23.0 | Away score before the pitch. |
| `balls` | Int64 | feature | 100% | 0.0 … 3.0 | Balls in the count BEFORE this pitch. |
| `batter` | Int64 | feature | 100% | 457705.0 … 828685.0 | Batter MLBAM id.<br>⚠ High-cardinality categorical. CatBoost handles natively; do not one-hot. |
| `fielder_2` | Int64 | feature | 100% | 506702.0 … 807083.0 | Catcher MLBAM id.<br>⚠ This is how catcher framing metrics join on. |
| `home_score` | Int64 | feature | 100% | 0.0 … 23.0 | Home score before the pitch. |
| `inning` | Int64 | feature | 100% | 1.0 … 13.0 | Inning number. |
| `inning_topbot` | str | feature | 100% | Top, Bot | Top or Bot half.<br>⚠ Top = away team batting. |
| `on_1b` | Int64 | feature | 31% | 457705.0 … 828599.0 | Runner id on first, null if empty.<br>⚠ Null means empty base, not missing data. |
| `on_2b` | Int64 | feature | 19% | 457705.0 … 828599.0 | Runner id on second, null if empty. |
| `on_3b` | Int64 | feature | 10% | 457705.0 … 828599.0 | Runner id on third, null if empty. |
| `outs_when_up` | Int64 | feature | 100% | 0.0 … 2.0 | Outs before this pitch. |
| `p_throws` | str | feature | 100% | L, R | Pitcher handedness, L or R. |
| `pfx_x` | Float64 | feature | 100% | -2.92 … 2.51 | Horizontal movement, ft. |
| `pfx_z` | Float64 | feature | 100% | -2.0 … 2.7 | Vertical movement, ft. |
| `pitch_type` | str | feature | 100% | SI, CH, FF | Pitch type code (FF, SL, CH, ...). |
| `pitcher` | Int64 | feature | 100% | 434378.0 … 837227.0 | Pitcher MLBAM id. |
| `plate_x` | Float64 | feature | 100% | -6.3508 … 7.0309 | Horizontal location at the plate, ft; 0 is the plate's centre.<br>⚠ Catcher's perspective. Sign flips meaning by handedness — use the absolute value or interact with `stand`. |
| `plate_z` | Float64 | feature | 100% | -5.0231 … 10.7025 | Height at the plate, ft above the ground. |
| `release_pos_x` | Float64 | feature | 100% | -4.9 … 5.08 | Horizontal release point, ft. |
| `release_pos_z` | Float64 | feature | 100% | 1.03 … 8.18 | Vertical release point, ft. |
| `release_speed` | Float64 | feature | 100% | 30.3 … 105.5 | Release velocity, mph. |
| `stand` | str | feature | 100% | R, L | Batter handedness, L or R. |
| `strikes` | Int64 | feature | 100% | 0.0 … 2.0 | Strikes in the count BEFORE this pitch. |
| `zone` | Int64 | feature | 100% | 1.0 … 14.0 | Savant's 1-14 zone bucket.<br>⚠ Savant's own zone grid, NOT the ABS zone. Coarse; prefer zone_dist_in. |
| `at_bat_number` | Int64 | identifier | 100% | 1.0 … 121.0 | Plate appearance number within the game, 1-based.<br>⚠ The Stats API's atBatIndex is 0-BASED. Join as at_bat_number - 1. |
| `game_pk` | Int64 | identifier | 100% | 822678.0 … 849851.0 | MLB game identifier. |
| `pitch_number` | Int64 | identifier | 100% | 1.0 … 18.0 | Pitch number within the plate appearance, 1-based. |
| `description` | str | metadata | 100% | called_strike, ball, hit_into_play | Pitch result as called.<br>⚠ called_strike / ball / foul / swinging_strike / hit_into_play / ... |
| `game_date` | datetime64[us] | metadata | 100% | 2026-09-26, 2026-09-27, 2026-09-16 | Date the game was played. |
| `sz_bot` | Float64 | metadata | 100% | 1.4678 … 1.7872 | Statcast's zone bottom, ft.<br>⚠ See sz_top. |
| `sz_top` | Float64 | metadata | 100% | 2.9084 … 3.5412 | Statcast's zone top, ft.<br>⚠ NOT the ABS zone. Stance-dependent and set per pitch. Using it for the ABS calculation biases every distance feature. |
| `type` | str | metadata | 100% | S, B, X | S, B or X. |
| `age_bat` | Int64 | UNDOCUMENTED | 100% | 19.0 … 40.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `age_bat_legacy` | Int64 | UNDOCUMENTED | 100% | 18.0 … 40.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `age_pit` | Int64 | UNDOCUMENTED | 100% | 21.0 … 43.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `age_pit_legacy` | Int64 | UNDOCUMENTED | 100% | 21.0 … 43.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `api_break_x_arm` | Float64 | UNDOCUMENTED | 100% | -2.92 … 2.76 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `api_break_x_batter_in` | Float64 | UNDOCUMENTED | 100% | -2.76 … 2.92 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `api_break_z_with_gravity` | Float64 | UNDOCUMENTED | 100% | 0.22 … 27.5 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `arm_angle` | Float64 | UNDOCUMENTED | 98% | -66.5 … 85.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `attack_angle` | Float64 | UNDOCUMENTED | 46% | -89.6099 … 88.466 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `attack_direction` | Float64 | UNDOCUMENTED | 46% | -179.8718 … 179.9709 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `away_team` | str | UNDOCUMENTED | 100% | NYM, PHI, LAA | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `ax` | Float64 | UNDOCUMENTED | 100% | -34.0323 … 29.8228 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `ay` | Float64 | UNDOCUMENTED | 100% | 2.7069 … 48.6278 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `az` | Float64 | UNDOCUMENTED | 100% | -50.3761 … 0.0001 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `babip_value` | Int64 | UNDOCUMENTED | 26% | 0.0 … 1.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `bat_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `bat_score_diff` | Int64 | UNDOCUMENTED | 100% | -22.0 … 22.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `bat_speed` | Float64 | UNDOCUMENTED | 46% | 0.3 … 88.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `bat_win_exp` | Float64 | UNDOCUMENTED | 100% | 0.001 … 1.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `batter_days_since_prev_game` | Int64 | UNDOCUMENTED | 98% | 0.0 … 144.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `batter_days_until_next_game` | Int64 | UNDOCUMENTED | 98% | 0.0 … 144.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `bb_type` | str | UNDOCUMENTED | 17% | popup, line_drive, ground_ball | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `break_angle_deprecated` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `break_length_deprecated` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `delta_home_win_exp` | Float64 | UNDOCUMENTED | 100% | -0.786 … 0.927 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `delta_pitcher_run_exp` | Float64 | UNDOCUMENTED | 100% | -2.703 … 0.617 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `delta_run_exp` | Float64 | UNDOCUMENTED | 100% | -0.617 … 2.703 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `des` | str | UNDOCUMENTED | 26% | Francisco Lindor pops out to shortstop Nasim N | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `effective_speed` | Float64 | UNDOCUMENTED | 100% | 26.5 … 107.8 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `estimated_ba_using_speedangle` | Float64 | UNDOCUMENTED | 17% | 0.001 … 1.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `estimated_slg_using_speedangle` | Float64 | UNDOCUMENTED | 17% | 0.001 … 4.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `estimated_woba_using_speedangle` | Float64 | UNDOCUMENTED | 25% | 0.0 … 2.044 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `events` | str | UNDOCUMENTED | 26% | field_out, single, strikeout | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_3` | Int64 | UNDOCUMENTED | 100% | 467793.0 … 828685.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_4` | Int64 | UNDOCUMENTED | 100% | 500743.0 … 823550.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_5` | Int64 | UNDOCUMENTED | 100% | 500743.0 … 828237.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_6` | Int64 | UNDOCUMENTED | 100% | 500743.0 … 828084.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_7` | Int64 | UNDOCUMENTED | 100% | 457705.0 … 823807.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_8` | Int64 | UNDOCUMENTED | 100% | 545361.0 … 828599.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fielder_9` | Int64 | UNDOCUMENTED | 100% | 457705.0 … 825761.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `fld_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `game_type` | str | UNDOCUMENTED | 100% | R, S, F | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `game_year` | Int64 | UNDOCUMENTED | 100% | 2026.0 … 2026.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `hc_x` | Float64 | UNDOCUMENTED | 17% | 2.81 … 246.92 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `hc_y` | Float64 | UNDOCUMENTED | 17% | 6.88 … 241.47 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `hit_distance_sc` | Int64 | UNDOCUMENTED | 33% | 0.0 … 474.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `hit_location` | Int64 | UNDOCUMENTED | 22% | 1.0 … 9.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `home_score_diff` | Int64 | UNDOCUMENTED | 100% | -21.0 … 22.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `home_team` | str | UNDOCUMENTED | 100% | WSH, TOR, TEX | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `home_win_exp` | Float64 | UNDOCUMENTED | 100% | 0.0 … 1.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `hyper_speed` | Float64 | UNDOCUMENTED | 33% | 88.0 … 119.7 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `if_fielding_alignment` | str | UNDOCUMENTED | 99% | Standard, Infield shade, Strategic | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `intercept_ball_minus_batter_pos_x_inches` | Float64 | UNDOCUMENTED | 46% | 4.0826 … 77.7269 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `intercept_ball_minus_batter_pos_y_inches` | Float64 | UNDOCUMENTED | 46% | -11.9796 … 71.9998 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `iso_value` | Int64 | UNDOCUMENTED | 26% | 0.0 … 3.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `launch_angle` | Int64 | UNDOCUMENTED | 33% | -90.0 … 90.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `launch_speed` | Float64 | UNDOCUMENTED | 33% | 2.3 … 119.7 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `launch_speed_angle` | Int64 | UNDOCUMENTED | 17% | 1.0 … 6.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `miss_distance` | Float64 | UNDOCUMENTED | 11% | 0.1 … 54.1668 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `n_priorpa_thisgame_player_at_bat` | Int64 | UNDOCUMENTED | 100% | 0.0 … 6.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `n_thruorder_pitcher` | Int64 | UNDOCUMENTED | 100% | 1.0 … 4.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `of_fielding_alignment` | str | UNDOCUMENTED | 99% | Standard, Strategic | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `pitch_name` | str | UNDOCUMENTED | 100% | Sinker, Changeup, 4-Seam Fastball | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `pitcher_days_since_prev_game` | Int64 | UNDOCUMENTED | 94% | 0.0 … 180.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `pitcher_days_until_next_game` | Int64 | UNDOCUMENTED | 93% | 0.0 … 180.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `player_name` | str | UNDOCUMENTED | 100% | Early, Connelly, Tong, Jonah, Irvin, Jake | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `post_away_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `post_bat_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `post_fld_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `post_home_score` | Int64 | UNDOCUMENTED | 100% | 0.0 … 23.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `release_extension` | Float64 | UNDOCUMENTED | 100% | 2.7 … 9.4 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `release_pos_y` | Float64 | UNDOCUMENTED | 100% | 51.05 … 57.8 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `release_spin_rate` | Int64 | UNDOCUMENTED | 99% | 2.0 … 3599.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `spin_axis` | Int64 | UNDOCUMENTED | 99% | 0.0 … 360.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `spin_dir` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `spin_rate_deprecated` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `sv_id` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `swing_length` | Float64 | UNDOCUMENTED | 46% | 0.3 … 14.1 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `swing_path_tilt` | Float64 | UNDOCUMENTED | 46% | 0.4288 … 89.9856 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `tfs_deprecated` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `tfs_zulu_deprecated` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `umpire` | Int64 | UNDOCUMENTED | 0% |  | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `vx0` | Float64 | UNDOCUMENTED | 100% | -18.9461 … 22.7517 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `vy0` | Float64 | UNDOCUMENTED | 100% | -153.3452 … -40.4707 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `vz0` | Float64 | UNDOCUMENTED | 100% | -19.5172 … 17.3877 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `woba_denom` | Int64 | UNDOCUMENTED | 25% | 0.0 … 1.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |
| `woba_value` | Float64 | UNDOCUMENTED | 26% | 0.0 … 2.0 | Not yet documented — a Statcast column this project does not use.<br>⚠ Look it up in Savant's CSV documentation and add it to REGISTRY in scripts/09_data_dictionary.py if you start using it. |

## MLB Stats API (script 02)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `is_overturned` | object | **target** | 1% | True, False | The challenge succeeded and the call was reversed.<br>⚠ The target for the OVERTURN model p(s). Null on unchallenged pitches by construction, not by data loss. |
| `was_challenged` | bool | **target** | 100% | True 1.1% | A challenge was actually made on this pitch.<br>⚠ The target for the BEHAVIOUR model (did they challenge?). Never a feature. |
| `challenge_call_code` | str | **LEAKAGE RISK** | 1% | C, B, F | Original call code on a challenged pitch. |
| `challenge_description` | str | **LEAKAGE RISK** | 1% | Called Strike, Ball, Foul | Free-text description of the review. |
| `challenger_id` | float64 | **LEAKAGE RISK** | 1% | 457705.0 … 823550.0 | Challenger's MLBAM id.<br>⚠ See challenger_role. |
| `challenger_role` | str | **LEAKAGE RISK** | 1% | catcher, batter, pitcher | batter, catcher or pitcher.<br>⚠ Known only once a challenge happened. Fine for splitting results by role; not an input to a model predicting whether a challenge occurs. |
| `challenging_side` | str | **LEAKAGE RISK** | 1% | home, away | 'home' or 'away'.<br>⚠ See challenger_role. |
| `hp_umpire` | str | feature | 100% | Austin Jones, Jen Pawol, Alfonso Márquez | Home-plate umpire name.<br>⚠ From the schedule endpoint with hydrate=officials. |
| `challenger_name` | str | identifier | 1% | Keibert Ruiz, Francisco Alvarez, Marcus Semien | Challenger's name. |
| `hp_umpire_id` | float64 | identifier | 100% | 427013.0 … 691042.0 | Home-plate umpire id. |

## Derived (script 08)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `would_be_overturned` | bool | **target** | 100% | True 2.6% | Counterfactual: the umpire's call disagrees with the ABS zone, so a challenge here would have succeeded.<br>⚠ LEAKAGE RISK AS A FEATURE. This is what lets you label unchallenged opportunities, which is the whole point of the denominator — but it is an outcome, so it belongs on the left-hand side of the model only. |
| `abs_strike` | bool | feature | 100% | True 47.3% | True if ABS would call this a strike. |
| `abs_zone_bot` | float64 | feature | 100% | 1.4625 … 1.7775 | ABS zone bottom = 27% of batter height, ft. |
| `abs_zone_top` | float64 | feature | 100% | 2.8979 … 3.5221 | ABS zone top = 53.5% of batter height, ft.<br>⚠ Falls back to sz_top when height is missing — check abs_zone_source. |
| `batting_side` | str | feature | 100% | away, home | 'home' or 'away' — which side is batting.<br>⚠ Derived from inning_topbot. |
| `called_strike` | bool | feature | 100% | True 16.1% | The umpire called this pitch a strike. |
| `challenges_remaining_batting` | int8 | feature | 100% | 0.0 … 2.0 | Challenges the batting side has BEFORE this pitch (0-2).<br>⚠ DERIVED, NOT PUBLISHED. Walked forward per game: start at 2, a failed challenge costs one, a successful one is retained, and in extra innings a side already at zero is granted one for that inning. This is the MDP's state variable r — hand-verify it (script 11). |
| `challenges_remaining_fielding` | int8 | feature | 100% | 0.0 … 2.0 | Challenges the fielding side has BEFORE this pitch (0-2).<br>⚠ See above. |
| `fielding_side` | str | feature | 100% | home, away | The side in the field. |
| `zone_dist_ft` | float64 | feature | 100% | -0.8292 … 7.9832 | Signed distance from the BALL'S SURFACE to the ABS zone, ft. Negative = overlapping the zone = ABS strike.<br>⚠ The ball's 1.45-inch radius is already subtracted, because any part of the ball touching the zone is a strike. |
| `zone_dist_in` | float64 | feature | 100% | -9.95 … 95.7983 | Same, in inches. The headline feature.<br>⚠ Highly non-linear near zero — give it a spline or let a tree split it. |
| `abs_zone_source` | str | metadata | 100% | batter_height, statcast_sz | 'batter_height' (correct) or 'statcast_sz' (fallback).<br>⚠ Any row with 'statcast_sz' has an approximated zone. Report the share in your limitations. |
| `is_called_pitch` | bool | metadata | 100% | True 51.8% | Taken pitch — called strike or ball. |
| `is_challenge_opportunity` | bool | metadata | 100% | True 49.8% | Either side had an opportunity.<br>⚠ Savant reports ~50% of pitches. |
| `is_challenge_opportunity_batting` | bool | metadata | 100% | True 15.5% | The batting side could have challenged: called strike, a challenge in hand, no position player pitching.<br>⚠ The model's denominator for batter behaviour. |
| `is_challenge_opportunity_fielding` | bool | metadata | 100% | True 34.3% | Same for the fielding side on a called ball. |
| `is_reasonable_pitch` | boolean | metadata | 100% | True 5.0% | Opportunity that was either genuinely wrong or within 3 inches of the edge.<br>⚠ A simplified stand-in for Savant's 'reasonable pitch' (~5%); theirs also uses RE288 run value. |
| `position_player_pitching` | bool | metadata | 100% | True 0.5% | A position player is on the mound.<br>⚠ Challenges are not available in these at-bats. |
| `sc_zone_dist_in` | float64 | metadata | 100% | -9.95 … 95.9879 | Distance using Statcast's zone instead, for robustness checks.<br>⚠ Comparison only. Do not model on it. |
| `zone_defs_disagree` | bool | metadata | 100% | True 0.3% | True where the ABS zone and Statcast's zone disagree.<br>⚠ Where the challenges live. Good diagnostic, weak feature. |

## MLB Stats API people (script 07)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `batter_height_ft` | float64 | feature | 100% | 5.4167 … 6.5833 | Listed batter height in feet.<br>⚠ Self-reported and rounded to the inch; a 1-inch error moves the zone edge about half an inch. |
| `pitcher_pos` | str | metadata | 100% | P, DH, 3B | Pitcher's listed primary position.<br>⚠ Used only to detect position players pitching. |

## MLB Stats API winProbability (script 06)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `away_win_prob` | float64 | feature | 100% | 0.0 … 100.0 | Away team win probability entering the PA. |
| `bat_win_prob` | float64 | feature | 100% | 0.0 … 100.0 | Win probability for the batting side.<br>⚠ This is V(s) for the side deciding whether to challenge. |
| `home_win_prob` | float64 | feature | 100% | 0.0 … 100.0 | Home team win probability entering the PA. |
| `leverage_index` | float64 | feature | 98% | 0.0 … 9.99 | Leverage index entering the plate appearance; 1.0 is average.<br>⚠ PLATE-APPEARANCE grain, not pitch. A challenge happens mid-PA, so this is the state entering the PA. |
| `home_win_prob_added` | float64 | metadata | 100% | -76.4 … 90.1 | WPA of the completed plate appearance.<br>⚠ LEAKAGE RISK if used as a feature — it describes how the PA ended. |

## Savant catcher framing (script 04)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `catcher_rv_11` | float64 | feature | 82% | -2.0 … 2.0 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |
| `catcher_rv_12` | float64 | feature | 82% | -3.0 … 3.0 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |
| `catcher_rv_13` | float64 | feature | 82% | -2.0 … 1.0 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |
| `catcher_rv_14` | float64 | feature | 82% | -4.0 … 6.0 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |
| `catcher_rv_16` | float64 | feature | 82% | -4.0 … 5.0 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |
| `catcher_rv_tot` | float64 | feature | 82% | -10.6989 … 8.8805 | Season framing metric for the catcher on this pitch.<br>⚠ Season-level, so it is constant within a catcher-season and partly reflects the same 2026 challenges you are modelling. |

## Derived umpire accuracy (script 05)

| Column | Type | Role | Non-null | Range / example | Definition |
|---|---|---|---:|---|---|
| `ump_accuracy` | float64 | feature | 100% | 0.9322 … 0.9623 | Season accuracy metric for the home-plate umpire.<br>⚠ Derived from the same pitches you are modelling — mild circularity. Consider leave-one-game-out if it ranks high in Week 8. |
| `ump_borderline_accuracy` | float64 | feature | 100% | 0.7188 … 0.8338 | Season accuracy metric for the home-plate umpire.<br>⚠ Derived from the same pitches you are modelling — mild circularity. Consider leave-one-game-out if it ranks high in Week 8. |
| `ump_missed_ball_rate` | float64 | feature | 100% | 0.0183 … 0.0515 | Season accuracy metric for the home-plate umpire.<br>⚠ Derived from the same pitches you are modelling — mild circularity. Consider leave-one-game-out if it ranks high in Week 8. |
| `ump_missed_strike_rate` | float64 | feature | 100% | 0.0142 … 0.0281 | Season accuracy metric for the home-plate umpire.<br>⚠ Derived from the same pitches you are modelling — mild circularity. Consider leave-one-game-out if it ranks high in Week 8. |
