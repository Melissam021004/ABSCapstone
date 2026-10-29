# Analytic Cohort — 2026

Frozen 2026-10-08T13:54 by `scripts/13_define_cohort.py`. Fingerprint `419af01ad9eee03f`.

**370,909 rows** from 720,209 pitches (51.5% retained), 7,897 of them challenged.

Changing anything here means re-running Weeks 7 onward. If you do change it, note the date and the reason — the fingerprint above is how you prove which cohort produced which result.

## Inclusion rules

| # | Rule | Rows kept | Dropped | Why |
|---:|---|---:|---:|---|
| 1 | `called_pitches_only` | 373,339 | 346,870 | A challenge can only follow a taken pitch. Swings are not challengeable, so they carry no information about the decision. |
| 2 | `drop_position_player_pitching` | 371,421 | 1,918 | Challenges are not available while a position player pitches, so these rows have a structurally impossible outcome. |
| 3 | `require_pitch_location` | 371,236 | 185 | Without a location there is no distance-from-zone feature, which is the model's primary input. |
| 4 | `require_abs_zone_from_height` | 370,909 | 327 | Rows falling back to Statcast's zone have a different zone definition. Keeping them would mix two measurement systems in one feature. |
| 5 | `require_challenge_inventory` | 370,909 | 0 | The inventory is the MDP's state variable; a row without it cannot be used for either the behaviour model or the policy. |
| 6 | `regular_season_only` | 370,909 | 0 | Postseason challenge behaviour may differ and the sample is small. Revisit in Spring if you want a postseason transportability test. |

## Temporal split

Split on **time, not at random**. The model predicts future challenges from past ones; a random split would let it train on September while being tested on September.

| Window | Dates | Pitches | Challenges |
|---|---|---:|---:|
| train | 2026-03-01 → 2026-07-31 | 252,016 | 5,219 |
| validate | 2026-08-01 → 2026-08-31 | 63,649 | 1,435 |
| test | 2026-09-01 → 2026-12-31 | 55,244 | 1,243 |

Tune on **validate** only. The **test** window is read exactly once, at the end of Week 8.

## Missingness

`structural` means the null is correct and meaningful, not a gap.

| Column | Null % | Treatment | Note |
|---|---:|---|---|
| `estimated_slg_using_speedangle` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `umpire` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `swing_length` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `bat_speed` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `attack_angle` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `attack_direction` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `swing_path_tilt` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `launch_speed_angle` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `intercept_ball_minus_batter_pos_x_inches` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `intercept_ball_minus_batter_pos_y_inches` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `estimated_ba_using_speedangle` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `launch_angle` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `spin_dir` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `hit_distance_sc` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `sv_id` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `launch_speed` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `tfs_zulu_deprecated` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `tfs_deprecated` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `spin_rate_deprecated` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `break_angle_deprecated` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `break_length_deprecated` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `bb_type` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `miss_distance` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `hc_x` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `hc_y` | 100.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `hyper_speed` | 99.99% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `challenger_role` | 97.88% | structural | Only exists once a challenge happened. |
| `challenger_id` | 97.88% | structural | Same. |
| `challenger_name` | 97.88% | structural | Same. |
| `challenging_side` | 97.88% | structural | Same. |
| `is_overturned` | 97.87% | structural | Null on unchallenged pitches by design. |
| `challenge_call_code` | 97.87% | structural | Same. |
| `challenge_description` | 97.87% | structural | Same. |
| `hit_location` | 97.34% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `estimated_woba_using_speedangle` | 93.2% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `des` | 93.19% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `woba_denom` | 93.12% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `events` | 93.11% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `babip_value` | 93.11% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `iso_value` | 93.11% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `woba_value` | 93.11% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `on_3b` | 90.95% | structural | Null means the base is empty. |
| `on_2b` | 81.72% | structural | Null means the base is empty. |
| `on_1b` | 69.07% | structural | Null means the base is empty. |
| `catcher_rv_tot` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `catcher_rv_16` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `catcher_rv_11` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `catcher_rv_12` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `catcher_rv_13` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `catcher_rv_14` | 18.38% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `pitcher_days_until_next_game` | 7.12% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `pitcher_days_since_prev_game` | 5.62% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `batter_days_until_next_game` | 2.29% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `batter_days_since_prev_game` | 1.99% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `leverage_index` | 1.75% | impute_or_flag | Plate-appearance grain; a few pitches may not match. Add a missing-indicator rather than dropping the row. |
| `arm_angle` | 1.15% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `hp_umpire` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `ump_accuracy` | 0.39% | impute_or_flag | Umpires with few games. Flag, don't drop. |
| `ump_borderline_accuracy` | 0.39% | impute_or_flag | Same. |
| `ump_missed_strike_rate` | 0.39% | impute_or_flag | Same. |
| `hp_umpire_id` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `ump_missed_ball_rate` | 0.39% | impute_or_flag | Same. |
| `home_win_prob_added` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `bat_win_prob` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `away_win_prob` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `home_win_prob` | 0.39% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `if_fielding_alignment` | 0.21% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `of_fielding_alignment` | 0.21% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `release_spin_rate` | 0.16% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `spin_axis` | 0.16% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `pitcher_pos` | 0.15% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `effective_speed` | 0.1% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `release_extension` | 0.09% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `delta_run_exp` | 0.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
| `delta_pitcher_run_exp` | 0.0% | DECIDE | Not yet classified — decide whether to drop, impute, or flag. |
