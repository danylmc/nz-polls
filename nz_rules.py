#!/usr/bin/env python3
"""
Laws of NZ Politics: Statistical Rules from 30 Years of Polling

Derives quantified "laws" of NZ politics from 1,016 polls (1993-2026),
economic data, and documented events. Tests incumbent fatigue, punctuated
equilibrium, cross-bloc tidal dynamics, leadership honeymoons, crisis
rallies, economic sentiment, election convergence, minor party lifecycles,
post-election reset effects, and the thermostatic model of public opinion.

Output: reports/nz_rules.md
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import statsmodels.api as sm

from analysis import load_all_polling_data
from events import ELECTION_DATES, INCUMBENTS, LEADERSHIP_CHANGES, CRISES
from economic_scraper import load_economic_data
from voter_flows import LEFT_BLOC, RIGHT_BLOC

warnings.filterwarnings("ignore")

REPORTS_DIR = Path(__file__).parent / "reports"

# Government periods: (label, party, start_date, end_date)
GOVERNMENTS = [
    ("Bolger/Shipley 1993-1999", "National", "1993-11-06", "1999-11-27"),
    ("Clark 1999-2008", "Labour", "1999-11-27", "2008-11-08"),
    ("Key/English 2008-2017", "National", "2008-11-08", "2017-10-26"),
    ("Ardern/Hipkins 2017-2023", "Labour", "2017-10-26", "2023-11-27"),
    ("Luxon 2023-", "National", "2023-11-27", "2026-12-31"),
]

# Term boundaries within governments
TERM_ELECTIONS = sorted(ELECTION_DATES.values())


# ============================================================================
# DATA PREPARATION HELPERS
# ============================================================================

def prepare_weekly_series(df):
    """Resample National and Labour polling to weekly frequency with linear interpolation."""
    ts = df[["date", "National", "Labour"]].dropna(subset=["National", "Labour"]).copy()
    ts = ts.set_index("date").sort_index()
    # Remove duplicate dates by averaging
    ts = ts.groupby(ts.index).mean()
    # Resample to weekly and interpolate
    weekly = ts.resample("W").mean().interpolate(method="linear")
    weekly = weekly.dropna()
    return weekly


def interpolate_economics_monthly():
    """Load annual economic CSV, cubic-spline interpolate to monthly."""
    econ_data = load_economic_data()
    if not econ_data:
        return pd.DataFrame()

    econ_df = pd.DataFrame(econ_data)
    # Assign to mid-year dates
    econ_df["date"] = pd.to_datetime(econ_df["year"].astype(str) + "-07-01")
    econ_df = econ_df.sort_values("date").set_index("date")

    result_frames = []
    for col in ["gdp_growth", "unemployment_rate", "cpi_inflation"]:
        series = econ_df[col].dropna()
        if len(series) < 4:
            continue
        # Cubic spline on numeric timestamps
        x = (series.index - series.index[0]).days.values.astype(float)
        y = series.values
        cs = CubicSpline(x, y)

        # Monthly dates spanning the range
        monthly_dates = pd.date_range(series.index[0], series.index[-1], freq="MS")
        monthly_x = (monthly_dates - series.index[0]).days.values.astype(float)
        monthly_vals = cs(monthly_x)

        result_frames.append(pd.DataFrame({col: monthly_vals}, index=monthly_dates))

    if not result_frames:
        return pd.DataFrame()

    result = pd.concat(result_frames, axis=1)
    result.index.name = "date"
    return result.reset_index()


def build_government_timeline(df):
    """Add government, months_in_office, and term_number columns."""
    df = df.copy()
    df["government"] = None
    df["gov_party"] = None
    df["months_in_office"] = np.nan
    df["term_number"] = np.nan

    for label, party, start, end in GOVERNMENTS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (df["date"] >= start_dt) & (df["date"] < end_dt)
        df.loc[mask, "government"] = label
        df.loc[mask, "gov_party"] = party
        df.loc[mask, "months_in_office"] = (
            (df.loc[mask, "date"] - start_dt).dt.days / 30.44
        )

        # Count which term this is for the party
        elections_before = [
            e for e in TERM_ELECTIONS
            if pd.Timestamp(e) >= start_dt and pd.Timestamp(e) < end_dt
        ]
        for i, elec in enumerate(elections_before):
            elec_dt = pd.Timestamp(elec)
            next_elec_dt = pd.Timestamp(elections_before[i + 1]) if i + 1 < len(elections_before) else end_dt
            term_mask = mask & (df["date"] >= elec_dt) & (df["date"] < next_elec_dt)
            df.loc[term_mask, "term_number"] = i + 1

    return df


def calculate_bloc_support(df):
    """Add left_bloc and right_bloc columns, zero-filling missing minor parties."""
    df = df.copy()
    for col in ["Green", "ACT", "NZ First"]:
        if col not in df.columns:
            df[col] = 0.0
    df["Green"] = df["Green"].fillna(0)
    df["ACT"] = df["ACT"].fillna(0)
    df["NZ First"] = df["NZ First"].fillna(0)

    df["left_bloc"] = df["Labour"].fillna(0) + df["Green"]
    df["right_bloc"] = df["National"].fillna(0) + df["ACT"] + df["NZ First"]
    return df


# ============================================================================
# ANALYSIS 1: INCUMBENT FATIGUE LAW
# ============================================================================

def analyze_incumbent_fatigue(df):
    """Test: governing party support erodes at a predictable rate per year."""
    print("  1. Incumbent Fatigue Law...")
    df_gov = build_government_timeline(df)

    results = {
        "per_government": [],
        "pooled": {},
    }

    all_months = []
    all_support = []
    all_gov_labels = []

    for label, party, start, end in GOVERNMENTS:
        mask = df_gov["government"] == label
        sub = df_gov.loc[mask].dropna(subset=["months_in_office", party])
        if len(sub) < 10:
            continue

        months = sub["months_in_office"].values
        support = sub[party].values

        slope, intercept, r, p, se = stats.linregress(months, support)
        pts_per_year = slope * 12

        results["per_government"].append({
            "government": label,
            "party": party,
            "n_polls": len(sub),
            "start_support": round(intercept, 1),
            "slope_per_month": round(slope, 3),
            "pts_per_year": round(pts_per_year, 2),
            "r": round(r, 3),
            "p_value": round(p, 6),
        })

        all_months.extend(months.tolist())
        all_support.extend(support.tolist())
        all_gov_labels.extend([label] * len(months))

    # Pooled regression with government fixed effects
    if len(all_months) > 50:
        pool_df = pd.DataFrame({
            "months": all_months,
            "support": all_support,
            "gov": all_gov_labels,
        })
        dummies = pd.get_dummies(pool_df["gov"], drop_first=True, dtype=float)
        X = pd.concat([pool_df[["months"]], dummies], axis=1)
        X = sm.add_constant(X)
        y = pool_df["support"]

        try:
            model = sm.OLS(y, X).fit()
            results["pooled"] = {
                "slope_per_month": round(model.params["months"], 4),
                "pts_per_year": round(model.params["months"] * 12, 2),
                "p_value": round(model.pvalues["months"], 6),
                "r_squared": round(model.rsquared, 3),
                "n": len(pool_df),
            }
        except Exception:
            pass

    return results


# ============================================================================
# ANALYSIS 2: PUNCTUATED EQUILIBRIUM
# ============================================================================

def analyze_punctuated_equilibrium(df):
    """Test: polling moves in sudden bursts, not gradual drift."""
    print("  2. Punctuated Equilibrium...")
    weekly = prepare_weekly_series(df)

    results = {"parties": {}}

    for party in ["National", "Labour"]:
        changes = weekly[party].diff().dropna().values
        abs_changes = np.abs(changes)
        total_movement = abs_changes.sum()

        if total_movement == 0 or len(changes) < 50:
            continue

        party_results = {
            "n_weeks": len(changes),
        }

        # 1. Movement concentration ratio
        sorted_abs = np.sort(abs_changes)[::-1]
        top10_pct = int(len(sorted_abs) * 0.1)
        top20_pct = int(len(sorted_abs) * 0.2)
        conc_10 = sorted_abs[:top10_pct].sum() / total_movement
        conc_20 = sorted_abs[:top20_pct].sum() / total_movement

        # Bootstrap p-value: how often does a Gaussian null produce this concentration?
        n_boot = 10000
        null_concs_10 = np.zeros(n_boot)
        rng = np.random.default_rng(42)
        gaussian_samples = rng.normal(0, changes.std(), size=(n_boot, len(changes)))
        for b in range(n_boot):
            abs_b = np.abs(gaussian_samples[b])
            sorted_b = np.sort(abs_b)[::-1]
            null_concs_10[b] = sorted_b[:top10_pct].sum() / abs_b.sum()

        boot_p = (null_concs_10 >= conc_10).mean()

        party_results["concentration"] = {
            "top_10pct_share": round(conc_10, 3),
            "top_20pct_share": round(conc_20, 3),
            "gaussian_null_mean": round(null_concs_10.mean(), 3),
            "bootstrap_p": round(boot_p, 4),
        }

        # 2. Markov regime-switching model
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            mod = MarkovRegression(
                changes, k_regimes=2, trend="c", switching_variance=True
            )
            res = mod.fit(disp=False)

            # Params are a numpy array: [p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]]
            param_names = res.model.param_names
            params_dict = dict(zip(param_names, res.params))

            sigma0 = np.sqrt(params_dict.get("sigma2[0]", res.params[-2]))
            sigma1 = np.sqrt(params_dict.get("sigma2[1]", res.params[-1]))

            if sigma0 < sigma1:
                stable_regime, shift_regime = 0, 1
                stable_sigma, shift_sigma = sigma0, sigma1
            else:
                stable_regime, shift_regime = 1, 0
                stable_sigma, shift_sigma = sigma1, sigma0

            # Fraction of time in shift regime (smoothed_marginal_probabilities is ndarray)
            smp = res.smoothed_marginal_probabilities
            shift_frac = (smp[:, shift_regime] > 0.5).mean()

            party_results["regime_switching"] = {
                "stable_sigma": round(float(stable_sigma), 3),
                "shift_sigma": round(float(shift_sigma), 3),
                "shift_regime_fraction": round(float(shift_frac), 3),
                "sigma_ratio": round(float(shift_sigma / stable_sigma), 1) if stable_sigma > 0 else None,
            }
        except Exception as e:
            party_results["regime_switching"] = {"error": str(e)[:100]}

        # 3. Kurtosis test
        kurt = stats.kurtosis(changes, fisher=True)
        kurt_stat, kurt_p = stats.kurtosistest(changes)

        party_results["kurtosis"] = {
            "excess_kurtosis": round(float(kurt), 2),
            "test_statistic": round(float(kurt_stat), 2),
            "p_value": round(float(kurt_p), 6),
            "leptokurtic": bool(kurt > 0 and kurt_p < 0.05),
        }

        results["parties"][party] = party_results

    return results


# ============================================================================
# ANALYSIS 3: CROSS-BLOC TIDAL DYNAMICS
# ============================================================================

def analyze_cross_bloc(df):
    """Test: left and right blocs move as coherent units."""
    print("  3. Cross-Bloc Tidal Dynamics...")
    df_bloc = calculate_bloc_support(df)
    df_bloc = df_bloc.dropna(subset=["National", "Labour"]).copy()

    results = {}

    # 1. Compare bloc-level vs party-level std
    party_stds = {
        "National_std": df_bloc["National"].std(),
        "Labour_std": df_bloc["Labour"].std(),
        "Green_std": df_bloc["Green"].std(),
        "ACT_std": df_bloc["ACT"].std(),
        "NZFirst_std": df_bloc["NZ First"].std(),
    }
    bloc_stds = {
        "left_bloc_std": df_bloc["left_bloc"].std(),
        "right_bloc_std": df_bloc["right_bloc"].std(),
    }

    results["volatility_comparison"] = {
        "party_level": {k: round(v, 2) for k, v in party_stds.items()},
        "bloc_level": {k: round(v, 2) for k, v in bloc_stds.items()},
        "left_bloc_smoother": bloc_stds["left_bloc_std"] < party_stds["Labour_std"],
        "right_bloc_smoother": bloc_stds["right_bloc_std"] < party_stds["National_std"],
    }

    # 2. Correlate left bloc changes with right bloc changes
    df_sorted = df_bloc.sort_values("date").copy()
    df_sorted["left_change"] = df_sorted["left_bloc"].diff()
    df_sorted["right_change"] = df_sorted["right_bloc"].diff()
    valid = df_sorted.dropna(subset=["left_change", "right_change"])

    if len(valid) > 20:
        corr, p = stats.pearsonr(valid["left_change"], valid["right_change"])
        results["bloc_change_correlation"] = {
            "pearson_r": round(corr, 3),
            "p_value": round(p, 6),
            "n": len(valid),
        }

    # 3. CUSUM on bloc difference (Right - Left)
    df_sorted["bloc_diff"] = df_sorted["right_bloc"] - df_sorted["left_bloc"]
    bloc_diff = df_sorted[["date", "bloc_diff"]].dropna()

    if len(bloc_diff) > 50:
        mean_diff = bloc_diff["bloc_diff"].mean()
        cusum = (bloc_diff["bloc_diff"] - mean_diff).cumsum().values

        # Identify sustained tidal periods: when CUSUM crosses zero
        sign_changes = np.diff(np.sign(cusum))
        n_crossings = np.count_nonzero(sign_changes)
        dates = bloc_diff["date"].values

        if n_crossings > 0:
            crossing_indices = np.where(sign_changes != 0)[0]
            # Calculate durations between crossings
            if len(crossing_indices) > 1:
                durations = []
                for i in range(len(crossing_indices) - 1):
                    d1 = pd.Timestamp(dates[crossing_indices[i]])
                    d2 = pd.Timestamp(dates[crossing_indices[i + 1]])
                    durations.append((d2 - d1).days)
                avg_tide_days = np.mean(durations)
            else:
                avg_tide_days = None
        else:
            avg_tide_days = None

        results["cusum_tides"] = {
            "mean_bloc_diff": round(mean_diff, 2),
            "n_tide_changes": int(n_crossings),
            "avg_tide_duration_days": round(avg_tide_days) if avg_tide_days else None,
            "current_cusum": round(float(cusum[-1]), 1),
            "cusum_direction": "Right-leaning" if cusum[-1] > 0 else "Left-leaning",
        }

    return results


# ============================================================================
# ANALYSIS 4: LEADERSHIP HONEYMOON LAW
# ============================================================================

def analyze_leadership_honeymoon(df):
    """Test: new leaders get a bounce that decays with a half-life."""
    print("  4. Leadership Honeymoon Law...")
    results = {"events": [], "summary": {}}

    mid_term_bounces = []
    election_bounces = []

    for event in LEADERSHIP_CHANGES:
        event_date = pd.Timestamp(event["date"])
        party = event["party"]
        is_midterm = event["type"] == "leader_change"

        # 60 days before
        before = df[(df["date"] >= event_date - pd.Timedelta(days=60)) &
                     (df["date"] < event_date)]
        # 30/60/90 days after
        windows = {}
        for days in [30, 60, 90]:
            after = df[(df["date"] > event_date) &
                       (df["date"] <= event_date + pd.Timedelta(days=days))]
            if len(after) >= 2 and party in after.columns:
                windows[f"after_{days}d"] = round(after[party].mean(), 1)

        if len(before) < 2 or party not in before.columns:
            continue

        before_mean = before[party].mean()
        if pd.isna(before_mean):
            continue

        entry = {
            "event": event["event"],
            "date": event["date"],
            "party": party,
            "type": "mid-term" if is_midterm else "election",
            "before_mean": round(before_mean, 1),
        }
        entry.update(windows)

        # Primary bounce = 30d after - before
        if "after_30d" in windows:
            bounce = windows["after_30d"] - before_mean
            entry["bounce_30d"] = round(bounce, 1)
            if is_midterm:
                mid_term_bounces.append(bounce)
            else:
                election_bounces.append(bounce)

        # Cross-party effect
        other_party = "National" if party == "Labour" else "Labour"
        other_before = before[other_party].mean()
        after_30 = df[(df["date"] > event_date) &
                      (df["date"] <= event_date + pd.Timedelta(days=30))]
        if len(after_30) >= 2:
            other_after = after_30[other_party].mean()
            if pd.notna(other_before) and pd.notna(other_after):
                entry["opponent_change_30d"] = round(other_after - other_before, 1)

        # Exponential decay fit (if enough data)
        after_90 = df[(df["date"] > event_date) &
                      (df["date"] <= event_date + pd.Timedelta(days=180))]
        if len(after_90) >= 8 and party in after_90.columns:
            t_vals = (after_90["date"] - event_date).dt.days.values.astype(float)
            y_vals = after_90[party].dropna().values
            t_vals = t_vals[:len(y_vals)]
            if len(t_vals) >= 8 and "bounce_30d" in entry and abs(entry["bounce_30d"]) > 1:
                try:
                    def decay_func(t, baseline, bounce, tau):
                        return baseline + bounce * np.exp(-t / tau)

                    p0 = [before_mean, entry["bounce_30d"], 60]
                    bounds = ([0, -30, 5], [80, 30, 365])
                    popt, _ = curve_fit(decay_func, t_vals, y_vals, p0=p0,
                                        bounds=bounds, maxfev=5000)
                    half_life = popt[2] * np.log(2)
                    entry["decay_half_life_days"] = round(half_life, 0)
                    entry["decay_tau"] = round(popt[2], 0)
                except Exception:
                    pass

        results["events"].append(entry)

    # Summary
    if mid_term_bounces:
        results["summary"]["midterm_avg_bounce"] = round(np.mean(mid_term_bounces), 1)
        results["summary"]["midterm_n"] = len(mid_term_bounces)
    if election_bounces:
        results["summary"]["election_avg_bounce"] = round(np.mean(election_bounces), 1)
        results["summary"]["election_n"] = len(election_bounces)

    all_bounces = mid_term_bounces + election_bounces
    if all_bounces:
        results["summary"]["overall_avg_bounce"] = round(np.mean(all_bounces), 1)

    return results


# ============================================================================
# ANALYSIS 5: CRISIS RALLY EFFECT
# ============================================================================

def analyze_crisis_rally(df):
    """Test: crises boost incumbent support for a measurable duration."""
    print("  5. Crisis Rally Effect...")
    results = {"events": []}

    for crisis in CRISES:
        crisis_date = pd.Timestamp(crisis["date"])

        # Determine incumbent at crisis time
        incumbent = None
        for year in sorted(ELECTION_DATES.keys()):
            if pd.Timestamp(ELECTION_DATES[year]) >= crisis_date:
                incumbent = INCUMBENTS.get(year)
                break

        if not incumbent:
            continue

        opposition = "Labour" if incumbent == "National" else "National"

        before = df[(df["date"] >= crisis_date - pd.Timedelta(days=60)) &
                     (df["date"] < crisis_date)]

        if len(before) < 2:
            continue

        inc_before = before[incumbent].mean()
        opp_before = before[opposition].mean()

        entry = {
            "event": crisis["event"],
            "date": crisis["date"],
            "incumbent": incumbent,
            "incumbent_before": round(inc_before, 1) if pd.notna(inc_before) else None,
        }

        # Check multiple windows
        for days in [30, 60, 90, 180]:
            after = df[(df["date"] > crisis_date) &
                       (df["date"] <= crisis_date + pd.Timedelta(days=days))]
            if len(after) >= 2:
                inc_after = after[incumbent].mean()
                opp_after = after[opposition].mean()
                if pd.notna(inc_after) and pd.notna(inc_before):
                    entry[f"incumbent_change_{days}d"] = round(inc_after - inc_before, 1)
                if pd.notna(opp_after) and pd.notna(opp_before):
                    entry[f"opposition_change_{days}d"] = round(opp_after - opp_before, 1)

        results["events"].append(entry)

    # Summary
    rally_effects = [e.get("incumbent_change_30d", 0) for e in results["events"]
                     if e.get("incumbent_change_30d") is not None]
    if rally_effects:
        results["summary"] = {
            "mean_incumbent_boost_30d": round(np.mean(rally_effects), 1),
            "n_crises": len(rally_effects),
        }

    return results


# ============================================================================
# ANALYSIS 6: ECONOMIC SENTIMENT (IMPROVED)
# ============================================================================

def analyze_economic_sentiment(df):
    """Test: economic conditions predict incumbent support at poll-level with lags."""
    print("  6. Economic Sentiment (improved)...")
    monthly_econ = interpolate_economics_monthly()

    if monthly_econ.empty:
        return {"error": "No economic data available"}

    results = {"lag_analysis": {}, "n_polls": 0}

    # For each poll, find the incumbent and match to economic data
    df_gov = build_government_timeline(df)
    df_gov = df_gov.dropna(subset=["gov_party"]).copy()

    # Get incumbent support per poll
    df_gov["inc_support"] = df_gov.apply(
        lambda r: r[r["gov_party"]] if pd.notna(r.get(r["gov_party"])) else np.nan,
        axis=1,
    )
    df_valid = df_gov.dropna(subset=["inc_support"]).copy()
    results["n_polls"] = len(df_valid)

    econ_vars = ["gdp_growth", "unemployment_rate", "cpi_inflation"]

    for lag_months in [0, 3, 6, 9, 12]:
        lag_results = {}
        for econ_var in econ_vars:
            if econ_var not in monthly_econ.columns:
                continue

            # Match each poll to the economic observation at (poll_date - lag)
            matched_support = []
            matched_econ = []
            gov_labels = []

            for _, row in df_valid.iterrows():
                poll_date = row["date"]
                lookup_date = poll_date - pd.DateOffset(months=lag_months)
                # Find nearest monthly econ observation
                diffs = (monthly_econ["date"] - lookup_date).abs()
                if diffs.min().days > 45:
                    continue
                nearest_idx = diffs.idxmin()
                econ_val = monthly_econ.loc[nearest_idx, econ_var]
                if pd.notna(econ_val):
                    matched_support.append(row["inc_support"])
                    matched_econ.append(econ_val)
                    gov_labels.append(row["government"])

            if len(matched_support) < 30:
                continue

            mdf = pd.DataFrame({
                "support": matched_support,
                "econ": matched_econ,
                "gov": gov_labels,
            })

            # Regression with government fixed effects
            dummies = pd.get_dummies(mdf["gov"], drop_first=True, dtype=float)
            X = pd.concat([mdf[["econ"]], dummies], axis=1)
            X = sm.add_constant(X)
            y = mdf["support"]

            try:
                model = sm.OLS(y, X).fit()
                lag_results[econ_var] = {
                    "coefficient": round(model.params["econ"], 3),
                    "p_value": round(model.pvalues["econ"], 4),
                    "n": len(mdf),
                    "significant": bool(model.pvalues["econ"] < 0.05),
                }
            except Exception:
                pass

        if lag_results:
            results["lag_analysis"][f"lag_{lag_months}m"] = lag_results

    # Also test 12-month changes in econ indicators
    results["change_analysis"] = {}
    for econ_var in econ_vars:
        if econ_var not in monthly_econ.columns:
            continue

        monthly_econ_sorted = monthly_econ.sort_values("date").copy()
        monthly_econ_sorted[f"{econ_var}_12m_change"] = (
            monthly_econ_sorted[econ_var] - monthly_econ_sorted[econ_var].shift(12)
        )

        matched_support = []
        matched_change = []
        gov_labels = []

        for _, row in df_valid.iterrows():
            poll_date = row["date"]
            diffs = (monthly_econ_sorted["date"] - poll_date).abs()
            if diffs.min().days > 45:
                continue
            nearest_idx = diffs.idxmin()
            change_val = monthly_econ_sorted.loc[nearest_idx, f"{econ_var}_12m_change"]
            if pd.notna(change_val):
                matched_support.append(row["inc_support"])
                matched_change.append(change_val)
                gov_labels.append(row["government"])

        if len(matched_support) < 30:
            continue

        mdf = pd.DataFrame({
            "support": matched_support,
            "econ_change": matched_change,
            "gov": gov_labels,
        })

        dummies = pd.get_dummies(mdf["gov"], drop_first=True, dtype=float)
        X = pd.concat([mdf[["econ_change"]], dummies], axis=1)
        X = sm.add_constant(X)
        y = mdf["support"]

        try:
            model = sm.OLS(y, X).fit()
            results["change_analysis"][econ_var] = {
                "coefficient": round(model.params["econ_change"], 3),
                "p_value": round(model.pvalues["econ_change"], 4),
                "n": len(mdf),
                "significant": bool(model.pvalues["econ_change"] < 0.05),
            }
        except Exception:
            pass

    return results


# ============================================================================
# ANALYSIS 7: ELECTION PROXIMITY BEHAVIOUR
# ============================================================================

def analyze_election_convergence(df):
    """Comprehensive analysis of how polling behaviour changes as elections approach."""
    print("  7. Election Proximity Behaviour...")
    results = {}

    df_valid = df.dropna(subset=["days_to_election", "National", "Labour"]).copy()
    df_valid = df_valid[df_valid["days_to_election"] >= 0].copy()

    BINS = [(0, 30), (30, 90), (90, 180), (180, 365), (365, 1200)]
    BIN_LABELS = ["0-30", "30-90", "90-180", "180-365", "365+"]

    def bin_label(dte):
        for (lo, hi), lbl in zip(BINS, BIN_LABELS):
            if lo <= dte < hi:
                return lbl
        return None

    df_valid["dte_bin"] = df_valid["days_to_election"].apply(bin_label)

    # --- 1. Polling frequency ---
    freq = {}
    for (lo, hi), lbl in zip(BINS, BIN_LABELS):
        sub = df_valid[(df_valid["days_to_election"] >= lo) & (df_valid["days_to_election"] < hi)]
        span_days = hi - lo
        rate = len(sub) / span_days * 30  # polls per month
        freq[lbl] = {"n": len(sub), "polls_per_month": round(rate, 1)}
    results["polling_frequency"] = freq

    # --- 2. Poll-to-poll volatility by proximity ---
    volatility = {}
    for party in ["National", "Labour"]:
        df_sorted = df_valid.sort_values("date").copy()
        df_sorted[f"abs_change"] = df_sorted[party].diff().abs()
        party_vol = {}
        for lbl in BIN_LABELS:
            sub = df_sorted[df_sorted["dte_bin"] == lbl]["abs_change"].dropna()
            if len(sub) >= 5:
                party_vol[lbl] = {
                    "mean_abs_change": round(sub.mean(), 2),
                    "median_abs_change": round(sub.median(), 2),
                    "n": len(sub),
                }
        volatility[party] = party_vol

        # Test final 30 vs rest
        late = df_sorted[df_sorted["days_to_election"] < 30]["abs_change"].dropna()
        early = df_sorted[df_sorted["days_to_election"] >= 90]["abs_change"].dropna()
        if len(late) > 10 and len(early) > 10:
            t, p = stats.ttest_ind(early, late)
            volatility[f"{party}_settling"] = {
                "early_mean": round(early.mean(), 2),
                "late_mean": round(late.mean(), 2),
                "t_stat": round(t, 2),
                "p_value": round(p, 4),
                "settles": bool(late.mean() < early.mean() and p < 0.05),
            }
    results["poll_volatility"] = volatility

    # --- 3. National-Labour gap ---
    df_valid["gap"] = (df_valid["National"] - df_valid["Labour"]).abs()
    gap_by_cycle = {}
    for year in sorted(df_valid["election_year"].unique()):
        cycle = df_valid[df_valid["election_year"] == year]
        if len(cycle) < 20:
            continue
        early_gap = cycle[cycle["days_to_election"] > 180]["gap"]
        late_gap = cycle[cycle["days_to_election"] <= 60]["gap"]
        if len(early_gap) > 3 and len(late_gap) > 3:
            gap_by_cycle[int(year)] = {
                "early_gap": round(early_gap.mean(), 1),
                "late_gap": round(late_gap.mean(), 1),
                "change": round(late_gap.mean() - early_gap.mean(), 1),
                "narrows": bool(late_gap.mean() < early_gap.mean()),
            }
    results["gap_by_cycle"] = gap_by_cycle
    if gap_by_cycle:
        narrows_count = sum(1 for v in gap_by_cycle.values() if v["narrows"])
        results["gap_summary"] = {
            "n_narrows": narrows_count,
            "n_widens": len(gap_by_cycle) - narrows_count,
            "n_cycles": len(gap_by_cycle),
        }

    # --- 4. Incumbent late fade ---
    df_gov = build_government_timeline(df_valid)
    incumbent_late = {}
    for year in sorted(df_valid["election_year"].unique()):
        cycle = df_gov[df_gov["election_year"] == year]
        inc = INCUMBENTS.get(year)
        if not inc or len(cycle) < 20:
            continue
        final_180 = cycle[cycle["days_to_election"] <= 180]
        if len(final_180) < 10:
            continue
        far = final_180[final_180["days_to_election"] > 90][inc]
        near = final_180[final_180["days_to_election"] <= 90][inc]
        if len(far) > 3 and len(near) > 3:
            incumbent_late[int(year)] = {
                "incumbent": inc,
                "far_mean": round(far.mean(), 1),
                "near_mean": round(near.mean(), 1),
                "change": round(near.mean() - far.mean(), 1),
                "fades": bool(near.mean() < far.mean()),
            }
    results["incumbent_late_fade"] = incumbent_late
    if incumbent_late:
        fades_count = sum(1 for v in incumbent_late.values() if v["fades"])
        results["incumbent_fade_summary"] = {
            "n_fades": fades_count,
            "n_rises": len(incumbent_late) - fades_count,
            "n_cycles": len(incumbent_late),
        }

    # --- 5. Minor party / NZ First late surge ---
    minor_proximity = {}
    for party in ["Green", "ACT", "NZ First"]:
        early = df_valid[df_valid["days_to_election"] > 180][party].dropna()
        late = df_valid[df_valid["days_to_election"] <= 30][party].dropna()
        if len(early) > 10 and len(late) > 10:
            t, p = stats.ttest_ind(early, late)
            minor_proximity[party] = {
                "early_mean": round(early.mean(), 1),
                "late_mean": round(late.mean(), 1),
                "change": round(late.mean() - early.mean(), 1),
                "p_value": round(p, 4),
                "significant": bool(p < 0.05),
            }
    results["minor_party_proximity"] = minor_proximity

    # --- 6. Total reported share ---
    df_valid["total_5party"] = df_valid[["National", "Labour", "Green", "ACT", "NZ First"]].fillna(0).sum(axis=1)
    total_by_bin = {}
    for lbl in BIN_LABELS:
        sub = df_valid[df_valid["dte_bin"] == lbl]["total_5party"]
        if len(sub) > 5:
            total_by_bin[lbl] = round(sub.mean(), 1)
    results["total_share_by_proximity"] = total_by_bin

    return results


# ============================================================================
# ANALYSIS 8: MINOR PARTY LIFECYCLE PATTERNS
# ============================================================================

def analyze_minor_party_lifecycle(df):
    """Test: minor parties are more volatile and act as dissatisfaction barometer."""
    print("  8. Minor Party Lifecycle...")
    results = {}

    df_bloc = calculate_bloc_support(df)

    # 1. Coefficient of variation comparison
    cv_results = {}
    for party in ["National", "Labour", "Green", "ACT", "NZ First"]:
        vals = df[party].dropna()
        if len(vals) > 10 and vals.mean() > 0:
            cv_results[party] = {
                "mean": round(vals.mean(), 2),
                "std": round(vals.std(), 2),
                "cv": round(vals.std() / vals.mean(), 3),
                "n": len(vals),
            }

    results["coefficient_of_variation"] = cv_results

    major_cvs = [cv_results[p]["cv"] for p in ["National", "Labour"] if p in cv_results]
    minor_cvs = [cv_results[p]["cv"] for p in ["Green", "ACT", "NZ First"] if p in cv_results]

    if major_cvs and minor_cvs:
        results["cv_comparison"] = {
            "major_avg_cv": round(np.mean(major_cvs), 3),
            "minor_avg_cv": round(np.mean(minor_cvs), 3),
            "minor_more_volatile": np.mean(minor_cvs) > np.mean(major_cvs),
        }

    # 2. Minor party total vs incumbent support
    df_gov = build_government_timeline(df)
    df_gov = df_gov.dropna(subset=["gov_party"]).copy()
    df_gov["total_minor"] = df_gov[["Green", "ACT", "NZ First"]].fillna(0).sum(axis=1)
    df_gov["inc_support"] = df_gov.apply(
        lambda r: r[r["gov_party"]] if pd.notna(r.get(r["gov_party"])) else np.nan,
        axis=1,
    )
    valid = df_gov.dropna(subset=["inc_support", "total_minor", "government"])

    if len(valid) > 30:
        dummies = pd.get_dummies(valid["government"], drop_first=True, dtype=float)
        X = pd.concat([valid[["inc_support"]], dummies], axis=1)
        X = sm.add_constant(X)
        y = valid["total_minor"]

        try:
            model = sm.OLS(y, X).fit()
            results["minor_vs_incumbent"] = {
                "coefficient": round(model.params["inc_support"], 3),
                "p_value": round(model.pvalues["inc_support"], 4),
                "counter_cyclical": bool(model.params["inc_support"] < 0 and model.pvalues["inc_support"] < 0.05),
                "n": len(valid),
            }
        except Exception:
            pass

    # 3. MMP threshold stability
    threshold_results = {}
    for party in ["Green", "ACT", "NZ First"]:
        vals = df[party].dropna()
        if len(vals) < 20:
            continue

        above_5 = (vals >= 5).sum()
        below_5 = (vals < 5).sum()
        pct_above = above_5 / len(vals) * 100

        # Count threshold crossings
        above_flag = (vals >= 5).values
        crossings = np.sum(np.abs(np.diff(above_flag.astype(int))))

        threshold_results[party] = {
            "pct_above_5": round(pct_above, 1),
            "n_crossings": int(crossings),
            "n_polls": len(vals),
        }

    results["mmp_threshold"] = threshold_results

    return results


# ============================================================================
# ANALYSIS 9: POST-ELECTION RESET
# ============================================================================

def analyze_post_election_reset(df):
    """Test: after elections, winner's support rises or drops."""
    print("  9. Post-Election Reset...")
    results = {"transitions": []}

    election_years = sorted(ELECTION_DATES.keys())

    for i, year in enumerate(election_years):
        if year >= 2026:
            continue

        election_date_str = ELECTION_DATES[year]
        election_date = pd.Timestamp(election_date_str)

        # Determine winner/loser
        incumbent = INCUMBENTS.get(year)
        if not incumbent:
            continue

        # Next cycle's incumbent is the winner
        next_year = election_years[i + 1] if i + 1 < len(election_years) else None
        if next_year:
            winner = INCUMBENTS.get(next_year)
        else:
            continue

        loser = "Labour" if winner == "National" else "National"

        # Last 5 polls before election
        pre_election = df[(df["date"] >= election_date - pd.Timedelta(days=30)) &
                          (df["date"] <= election_date)]

        # First 5 polls of next cycle
        post_election = df[(df["date"] > election_date) &
                           (df["date"] <= election_date + pd.Timedelta(days=120))]
        if len(post_election) > 5:
            post_election = post_election.head(5)

        if len(pre_election) < 2 or len(post_election) < 2:
            continue

        entry = {"election": year, "winner": winner, "loser": loser}

        for role, party in [("winner", winner), ("loser", loser)]:
            pre_mean = pre_election[party].mean()
            post_mean = post_election[party].mean()
            if pd.notna(pre_mean) and pd.notna(post_mean):
                entry[f"{role}_pre"] = round(pre_mean, 1)
                entry[f"{role}_post"] = round(post_mean, 1)
                entry[f"{role}_change"] = round(post_mean - pre_mean, 1)

        results["transitions"].append(entry)

    # Summary
    winner_changes = [t["winner_change"] for t in results["transitions"]
                      if "winner_change" in t]
    loser_changes = [t["loser_change"] for t in results["transitions"]
                     if "loser_change" in t]

    if winner_changes:
        results["summary"] = {
            "avg_winner_change": round(np.mean(winner_changes), 1),
            "avg_loser_change": round(np.mean(loser_changes), 1) if loser_changes else None,
            "n_transitions": len(winner_changes),
            "winner_boost_count": sum(1 for c in winner_changes if c > 0),
            "winner_drop_count": sum(1 for c in winner_changes if c < 0),
        }

    return results


# ============================================================================
# ANALYSIS 10: THERMOSTATIC MODEL
# ============================================================================

def analyze_thermostatic(df):
    """Test: public opinion acts as a thermostat — when governing party support
    rises 'too high', it subsequently corrects downward, and vice versa.

    Wlezien (1995): the public adjusts preferences in response to policy/government
    position, creating negative feedback. We test:
    1. Error-correction: does deviation from long-run mean predict subsequent change?
    2. Asymmetric correction: is downward correction (from highs) faster than upward?
    3. Cross-party thermostat: does high incumbent support predict opposition gains?
    """
    print("  10. Thermostatic Model...")
    results = {}

    df_gov = build_government_timeline(df)

    for party in ["National", "Labour"]:
        party_data = df_gov[["date", party, "government"]].dropna(subset=[party]).copy()
        party_data = party_data.sort_values("date")

        if len(party_data) < 50:
            continue

        party_results = {}

        # 1. Error-correction model within each government
        # For each government period, compute deviation from government-period mean
        # Test: Δsupport_t = α + β*(support_{t-1} - mean) + ε
        # Negative β = thermostatic correction
        all_deviations = []
        all_changes = []

        for gov_label in party_data["government"].dropna().unique():
            gov_mask = party_data["government"] == gov_label
            gov_data = party_data[gov_mask].copy()
            if len(gov_data) < 10:
                continue

            gov_mean = gov_data[party].mean()
            gov_data["deviation"] = gov_data[party] - gov_mean
            gov_data["change"] = gov_data[party].diff()
            gov_data = gov_data.dropna(subset=["deviation", "change"])

            # Lag deviation by 1
            gov_data["lagged_deviation"] = gov_data["deviation"].shift(1)
            gov_data = gov_data.dropna(subset=["lagged_deviation"])

            all_deviations.extend(gov_data["lagged_deviation"].tolist())
            all_changes.extend(gov_data["change"].tolist())

        if len(all_deviations) > 30:
            slope, intercept, r, p, se = stats.linregress(all_deviations, all_changes)
            party_results["error_correction"] = {
                "beta": round(slope, 4),
                "r": round(r, 3),
                "p_value": round(p, 6),
                "n": len(all_deviations),
                "thermostatic": bool(slope < 0 and p < 0.05),
                "interpretation": (
                    f"Each 1pt above mean predicts {abs(slope):.2f}pt correction next poll"
                    if slope < 0 and p < 0.05
                    else "No significant thermostatic correction"
                ),
            }

        # 2. Asymmetric correction: split deviations into positive (above mean) and negative
        if len(all_deviations) > 30:
            dev_arr = np.array(all_deviations)
            chg_arr = np.array(all_changes)

            above_mask = dev_arr > 0
            below_mask = dev_arr < 0

            asym = {}
            if above_mask.sum() > 10:
                s_above, _, _, p_above, _ = stats.linregress(dev_arr[above_mask], chg_arr[above_mask])
                asym["above_mean_beta"] = round(s_above, 4)
                asym["above_mean_p"] = round(p_above, 4)
                asym["above_mean_n"] = int(above_mask.sum())

            if below_mask.sum() > 10:
                s_below, _, _, p_below, _ = stats.linregress(dev_arr[below_mask], chg_arr[below_mask])
                asym["below_mean_beta"] = round(s_below, 4)
                asym["below_mean_p"] = round(p_below, 4)
                asym["below_mean_n"] = int(below_mask.sum())

            if "above_mean_beta" in asym and "below_mean_beta" in asym:
                asym["correction_asymmetric"] = bool(
                    abs(asym["above_mean_beta"]) != abs(asym["below_mean_beta"])
                )
                asym["faster_correction"] = (
                    "from above" if abs(asym.get("above_mean_beta", 0)) > abs(asym.get("below_mean_beta", 0))
                    else "from below"
                )

            party_results["asymmetric_correction"] = asym

        results[party] = party_results

    # 3. Cross-party thermostat: high incumbent → opposition gains
    df_gov2 = df_gov.dropna(subset=["gov_party", "National", "Labour"]).copy()
    df_gov2 = df_gov2.sort_values("date")
    df_gov2["inc_support"] = df_gov2.apply(
        lambda r: r[r["gov_party"]] if pd.notna(r.get(r["gov_party"])) else np.nan,
        axis=1,
    )
    opp_party_map = {"National": "Labour", "Labour": "National"}
    df_gov2["opp_support"] = df_gov2.apply(
        lambda r: r[opp_party_map.get(r["gov_party"], "")] if pd.notna(r.get(opp_party_map.get(r["gov_party"], ""))) else np.nan,
        axis=1,
    )
    df_gov2 = df_gov2.dropna(subset=["inc_support", "opp_support"]).copy()
    df_gov2["opp_change"] = df_gov2["opp_support"].diff()
    df_gov2["inc_lagged"] = df_gov2["inc_support"].shift(1)
    df_gov2 = df_gov2.dropna(subset=["opp_change", "inc_lagged"])

    # Compute deviation of incumbent from government-period mean
    all_inc_devs = []
    all_opp_changes = []
    for gov_label in df_gov2["government"].dropna().unique():
        gm = df_gov2[df_gov2["government"] == gov_label]
        if len(gm) < 10:
            continue
        gov_inc_mean = gm["inc_support"].mean()
        deviations = gm["inc_lagged"] - gov_inc_mean
        all_inc_devs.extend(deviations.tolist())
        all_opp_changes.extend(gm["opp_change"].tolist())

    if len(all_inc_devs) > 30:
        slope, intercept, r, p, se = stats.linregress(all_inc_devs, all_opp_changes)
        results["cross_party_thermostat"] = {
            "beta": round(slope, 4),
            "r": round(r, 3),
            "p_value": round(p, 6),
            "n": len(all_inc_devs),
            "interpretation": (
                f"High incumbent support predicts opposition gains (β={slope:.3f})"
                if slope > 0 and p < 0.05
                else "No significant cross-party thermostatic effect"
            ),
        }

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def strength_label(p_value=None, r=None, n=None, confirmed=None):
    """Assign a strength rating based on statistical evidence."""
    if confirmed is False:
        return "Not supported"
    if p_value is not None and p_value < 0.001 and n and n > 100:
        return "Strong"
    if p_value is not None and p_value < 0.01:
        return "Strong"
    if p_value is not None and p_value < 0.05:
        return "Moderate"
    if n and n < 10:
        return "Suggestive"
    return "Suggestive"


def generate_report(all_results):
    """Generate the full markdown report."""
    lines = []
    lines.append("# Laws of NZ Politics: Statistical Rules from 30 Years of Polling")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*")
    lines.append("")

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | Rule | Statement | Key Statistic | Strength |")
    lines.append("|---|------|-----------|---------------|----------|")

    summaries = _build_summaries(all_results)
    for s in summaries:
        lines.append(f"| {s['num']} | {s['name']} | {s['statement']} | {s['stat']} | {s['strength']} |")

    lines.append("")

    # ---------------------------------------------------------------
    # Detailed findings
    # ---------------------------------------------------------------
    lines.append("## Detailed Findings")
    lines.append("")

    _write_fatigue_section(lines, all_results.get("incumbent_fatigue", {}))
    _write_punctuated_section(lines, all_results.get("punctuated_equilibrium", {}))
    _write_bloc_section(lines, all_results.get("cross_bloc", {}))
    _write_honeymoon_section(lines, all_results.get("leadership_honeymoon", {}))
    _write_crisis_section(lines, all_results.get("crisis_rally", {}))
    _write_economic_section(lines, all_results.get("economic_sentiment", {}))
    _write_convergence_section(lines, all_results.get("election_convergence", {}))
    _write_minor_section(lines, all_results.get("minor_party_lifecycle", {}))
    _write_reset_section(lines, all_results.get("post_election_reset", {}))
    _write_thermostatic_section(lines, all_results.get("thermostatic", {}))

    # ---------------------------------------------------------------
    # Methodology
    # ---------------------------------------------------------------
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Data**: 1,016 party-vote polls scraped from Wikipedia (1993-2026)")
    lines.append("- **Economic data**: World Bank API (GDP growth, unemployment, CPI), annual 1960-2024,")
    lines.append("  cubic-spline interpolated to monthly for poll-level matching")
    lines.append("- **Weekly series**: National and Labour resampled to weekly frequency via linear interpolation")
    lines.append("- **Government fixed effects**: All regressions include government-period dummies to avoid")
    lines.append("  confounding between-government variation with within-government trends")
    lines.append("- **Regime switching**: 2-regime Markov model (statsmodels) with switching variance")
    lines.append("- **Bootstrap**: 10,000 resamples for movement concentration null distribution")
    lines.append("")

    lines.append("## Data Limitations")
    lines.append("")
    lines.append("- **Polling accuracy**: polls have house effects and sampling error (~3% margin)")
    lines.append("- **Annual economic data**: spline interpolation adds smoothness that may not reflect")
    lines.append("  within-year economic shocks")
    lines.append("- **Small N for events**: only 4 mid-term leadership changes, 4 crises, ~11 elections")
    lines.append("- **Survivorship in minor parties**: parties that disappeared (Alliance, United Future)")
    lines.append("  are not tracked, biasing lifecycle analysis")
    lines.append("- **No causal identification**: all findings are correlational")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by nz_rules.py*")

    return "\n".join(lines)


def _build_summaries(R):
    """Build one-line summaries for the summary table."""
    summaries = []

    # 1. Incumbent fatigue
    fatigue = R.get("incumbent_fatigue", {})
    pooled = fatigue.get("pooled", {})
    if pooled:
        summaries.append({
            "num": 1, "name": "Incumbent Fatigue",
            "statement": f"Governing parties lose ~{abs(pooled.get('pts_per_year', 0)):.1f} points/year",
            "stat": f"slope={pooled.get('pts_per_year', '?')} pts/yr, p={pooled.get('p_value', '?')}",
            "strength": strength_label(pooled.get("p_value"), n=pooled.get("n")),
        })
    else:
        summaries.append({"num": 1, "name": "Incumbent Fatigue", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 2. Punctuated equilibrium
    pe = R.get("punctuated_equilibrium", {}).get("parties", {})
    nat_pe = pe.get("National", {})
    conc = nat_pe.get("concentration", {})
    if conc:
        summaries.append({
            "num": 2, "name": "Punctuated Equilibrium",
            "statement": f"Top 10% of weeks account for {conc.get('top_10pct_share', 0)*100:.0f}% of movement",
            "stat": f"bootstrap p={conc.get('bootstrap_p', '?')}, kurtosis={nat_pe.get('kurtosis', {}).get('excess_kurtosis', '?')}",
            "strength": strength_label(conc.get("bootstrap_p")),
        })
    else:
        summaries.append({"num": 2, "name": "Punctuated Equilibrium", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 3. Cross-bloc tidal
    bloc = R.get("cross_bloc", {})
    bloc_corr = bloc.get("bloc_change_correlation", {})
    if bloc_corr:
        summaries.append({
            "num": 3, "name": "Cross-Bloc Tides",
            "statement": f"Left/right blocs move inversely (r={bloc_corr.get('pearson_r', '?')})",
            "stat": f"r={bloc_corr.get('pearson_r', '?')}, p={bloc_corr.get('p_value', '?')}",
            "strength": strength_label(bloc_corr.get("p_value"), n=bloc_corr.get("n")),
        })
    else:
        summaries.append({"num": 3, "name": "Cross-Bloc Tides", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 4. Leadership honeymoon
    honey = R.get("leadership_honeymoon", {}).get("summary", {})
    if honey:
        summaries.append({
            "num": 4, "name": "Leadership Honeymoon",
            "statement": f"New leaders get ~{honey.get('overall_avg_bounce', '?')} point bounce",
            "stat": f"midterm avg={honey.get('midterm_avg_bounce', '?')}, election avg={honey.get('election_avg_bounce', '?')}",
            "strength": "Suggestive" if honey.get("midterm_n", 0) + honey.get("election_n", 0) < 10 else "Moderate",
        })
    else:
        summaries.append({"num": 4, "name": "Leadership Honeymoon", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 5. Crisis rally
    crisis = R.get("crisis_rally", {}).get("summary", {})
    if crisis:
        summaries.append({
            "num": 5, "name": "Crisis Rally",
            "statement": f"Incumbent gains ~{crisis.get('mean_incumbent_boost_30d', '?')} points in 30 days",
            "stat": f"mean boost={crisis.get('mean_incumbent_boost_30d', '?')}, n={crisis.get('n_crises', '?')}",
            "strength": "Suggestive",
        })
    else:
        summaries.append({"num": 5, "name": "Crisis Rally", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 6. Economic sentiment
    econ = R.get("economic_sentiment", {})
    any_sig = False
    best_econ_stat = "No significant effects"
    for lag_key, lag_data in econ.get("lag_analysis", {}).items():
        for var, vdata in lag_data.items():
            if vdata.get("significant"):
                any_sig = True
                best_econ_stat = f"{var} at {lag_key}: coef={vdata['coefficient']}, p={vdata['p_value']}"
                break
        if any_sig:
            break
    for var, vdata in econ.get("change_analysis", {}).items():
        if vdata.get("significant"):
            any_sig = True
            best_econ_stat = f"{var} 12m change: coef={vdata['coefficient']}, p={vdata['p_value']}"
            break

    summaries.append({
        "num": 6, "name": "Economic Sentiment",
        "statement": "Economic conditions predict incumbent support" if any_sig else "Polls insensitive to economics",
        "stat": best_econ_stat,
        "strength": "Moderate" if any_sig else "Strong (null)",
    })

    # 7. Election proximity
    conv = R.get("election_convergence", {})
    fade_sum = conv.get("incumbent_fade_summary", {})
    lab_settle = conv.get("poll_volatility", {}).get("Labour_settling", {})
    nzf = conv.get("minor_party_proximity", {}).get("NZ First", {})
    parts = []
    if fade_sum:
        parts.append(f"incumbent fades in {fade_sum.get('n_fades','?')}/{fade_sum.get('n_cycles','?')} elections")
    if lab_settle and lab_settle.get("settles"):
        parts.append("Labour settles late")
    if nzf and nzf.get("significant") and nzf.get("change", 0) > 0:
        parts.append(f"NZF surges +{nzf['change']}pts")
    gap_sum = conv.get("gap_summary", {})
    if gap_sum:
        parts.append(f"gap narrows in only {gap_sum.get('n_narrows','?')}/{gap_sum.get('n_cycles','?')} cycles")
    if parts:
        summaries.append({
            "num": 7, "name": "Election Proximity",
            "statement": "; ".join(parts),
            "stat": f"incumbent fade {fade_sum.get('n_fades','?')}/{fade_sum.get('n_cycles','?')}, NZF late +{nzf.get('change','?')}pts p={nzf.get('p_value','?')}",
            "strength": "Moderate",
        })
    else:
        summaries.append({"num": 7, "name": "Election Proximity", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 8. Minor party lifecycle
    minor = R.get("minor_party_lifecycle", {})
    cv_comp = minor.get("cv_comparison", {})
    if cv_comp:
        summaries.append({
            "num": 8, "name": "Minor Party Volatility",
            "statement": f"Minor parties {cv_comp.get('minor_avg_cv', 0)/cv_comp.get('major_avg_cv', 1):.1f}x more volatile (CV) than major",
            "stat": f"major CV={cv_comp.get('major_avg_cv', '?')}, minor CV={cv_comp.get('minor_avg_cv', '?')}",
            "strength": "Moderate",
        })
    else:
        summaries.append({"num": 8, "name": "Minor Party Volatility", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 9. Post-election reset
    reset = R.get("post_election_reset", {}).get("summary", {})
    if reset:
        summaries.append({
            "num": 9, "name": "Post-Election Reset",
            "statement": f"Winners move {reset.get('avg_winner_change', '?'):+.1f} pts, losers {reset.get('avg_loser_change', '?'):+.1f} pts",
            "stat": f"n={reset.get('n_transitions', '?')} transitions",
            "strength": "Suggestive",
        })
    else:
        summaries.append({"num": 9, "name": "Post-Election Reset", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    # 10. Thermostatic model
    thermo = R.get("thermostatic", {})
    nat_ec = thermo.get("National", {}).get("error_correction", {})
    lab_ec = thermo.get("Labour", {}).get("error_correction", {})
    if nat_ec or lab_ec:
        # Use whichever party has data
        ec = nat_ec if nat_ec else lab_ec
        party_name = "National" if nat_ec else "Labour"
        thermostatic_confirmed = ec.get("thermostatic", False)
        summaries.append({
            "num": 10, "name": "Thermostatic Model",
            "statement": f"Deviations self-correct (β={ec.get('beta', '?')})" if thermostatic_confirmed
                         else "No significant thermostatic correction",
            "stat": f"{party_name} β={ec.get('beta', '?')}, p={ec.get('p_value', '?')}, n={ec.get('n', '?')}",
            "strength": strength_label(ec.get("p_value"), n=ec.get("n")),
        })
    else:
        summaries.append({"num": 10, "name": "Thermostatic Model", "statement": "Insufficient data", "stat": "N/A", "strength": "N/A"})

    return summaries


# --- Section writers ---

def _write_fatigue_section(lines, r):
    lines.append("### Rule 1: Incumbent Fatigue Law")
    lines.append("")

    per_gov = r.get("per_government", [])
    if per_gov:
        lines.append("**Statement:** Governing parties lose support at a measurable rate per year in office.")
        lines.append("")
        lines.append("| Government | Party | N | Start % | Pts/Year | r | p-value |")
        lines.append("|------------|-------|---|---------|----------|---|---------|")
        for g in per_gov:
            lines.append(
                f"| {g['government']} | {g['party']} | {g['n_polls']} | {g['start_support']} | "
                f"{g['pts_per_year']:+.2f} | {g['r']:.3f} | {g['p_value']:.4f} |"
            )
        lines.append("")

    pooled = r.get("pooled", {})
    if pooled:
        lines.append(f"**Pooled estimate (with government fixed effects):** {pooled['pts_per_year']:+.2f} points/year "
                      f"(p = {pooled['p_value']:.4f}, R² = {pooled['r_squared']:.3f}, n = {pooled['n']})")
        lines.append("")

    lines.append("**Caveats:** Fatigue rate varies across governments; external shocks (COVID, crises) ")
    lines.append("confound the time trend. Fixed effects absorb level differences but not within-government shocks.")
    lines.append("")


def _write_punctuated_section(lines, r):
    lines.append("### Rule 2: Punctuated Equilibrium")
    lines.append("")
    lines.append("**Statement:** Polling moves in sudden bursts, not gradual drift. Most weeks are stable;")
    lines.append("a small fraction of weeks accounts for most total movement.")
    lines.append("")

    for party, data in r.get("parties", {}).items():
        lines.append(f"**{party}** ({data.get('n_weeks', '?')} weeks):")
        lines.append("")

        conc = data.get("concentration", {})
        if conc:
            lines.append(f"- Movement concentration: top 10% of weeks = {conc['top_10pct_share']*100:.0f}% of total |Δ| "
                          f"(Gaussian null: {conc['gaussian_null_mean']*100:.0f}%, bootstrap p = {conc['bootstrap_p']:.4f})")
            lines.append(f"- Top 20% of weeks = {conc['top_20pct_share']*100:.0f}% of total movement")

        rs = data.get("regime_switching", {})
        if rs and "error" not in rs:
            lines.append(f"- Regime switching: stable σ = {rs['stable_sigma']:.2f}, shift σ = {rs['shift_sigma']:.2f} "
                          f"(ratio: {rs['sigma_ratio']}x)")
            lines.append(f"- Fraction of time in shift regime: {rs['shift_regime_fraction']*100:.0f}%")

        kurt = data.get("kurtosis", {})
        if kurt:
            lines.append(f"- Excess kurtosis: {kurt['excess_kurtosis']:.2f} (p = {kurt['p_value']:.4f}) — "
                          f"{'leptokurtic (fat-tailed bursts confirmed)' if kurt['leptokurtic'] else 'not significantly leptokurtic'}")

        lines.append("")


def _write_bloc_section(lines, r):
    lines.append("### Rule 3: Cross-Bloc Tidal Dynamics")
    lines.append("")
    lines.append("**Statement:** Left (Labour+Green) and right (National+ACT+NZFirst) blocs move as")
    lines.append("coherent opposing units; bloc-level support is smoother than party-level.")
    lines.append("")

    vol = r.get("volatility_comparison", {})
    if vol:
        lines.append("**Volatility comparison (std dev):**")
        lines.append("")
        party_stds = vol.get("party_level", {})
        bloc_stds = vol.get("bloc_level", {})
        for k, v in party_stds.items():
            lines.append(f"- {k}: {v}")
        for k, v in bloc_stds.items():
            lines.append(f"- {k}: {v}")
        lines.append(f"- Left bloc smoother than Labour alone: {vol.get('left_bloc_smoother', '?')}")
        lines.append(f"- Right bloc smoother than National alone: {vol.get('right_bloc_smoother', '?')}")
        lines.append("")

    bc = r.get("bloc_change_correlation", {})
    if bc:
        lines.append(f"**Bloc change correlation:** r = {bc['pearson_r']:.3f} (p = {bc['p_value']:.6f}, n = {bc['n']})")
        lines.append("")

    cusum = r.get("cusum_tides", {})
    if cusum:
        lines.append("**CUSUM tide analysis:**")
        lines.append(f"- Mean bloc difference (Right - Left): {cusum['mean_bloc_diff']:+.1f} points")
        lines.append(f"- Number of tide changes: {cusum['n_tide_changes']}")
        if cusum.get("avg_tide_duration_days"):
            lines.append(f"- Average tide duration: {cusum['avg_tide_duration_days']} days ({cusum['avg_tide_duration_days']/365:.1f} years)")
        lines.append(f"- Current direction: {cusum['cusum_direction']}")
        lines.append("")

    lines.append("**Caveats:** Bloc composition is fixed (NZ First classified as right); in practice")
    lines.append("NZ First acts as a swing party. Green party data sparse pre-2002.")
    lines.append("")


def _write_honeymoon_section(lines, r):
    lines.append("### Rule 4: Leadership Honeymoon Law")
    lines.append("")
    lines.append("**Statement:** New leaders receive a polling bounce that decays over time.")
    lines.append("Mid-term changes produce larger bounces than election transitions.")
    lines.append("")

    events = r.get("events", [])
    if events:
        lines.append("| Event | Type | Before | After 30d | Bounce | Opponent Δ | Decay half-life |")
        lines.append("|-------|------|--------|-----------|--------|------------|-----------------|")
        for e in events:
            bounce = e.get("bounce_30d", "—")
            bounce_str = f"{bounce:+.1f}" if isinstance(bounce, (int, float)) else bounce
            opp = e.get("opponent_change_30d", "—")
            opp_str = f"{opp:+.1f}" if isinstance(opp, (int, float)) else opp
            hl = e.get("decay_half_life_days", "—")
            hl_str = f"{hl:.0f}d" if isinstance(hl, (int, float)) else hl
            lines.append(
                f"| {e['event'][:35]} | {e['type']} | {e['before_mean']} | "
                f"{e.get('after_30d', '—')} | {bounce_str} | {opp_str} | {hl_str} |"
            )
        lines.append("")

    summary = r.get("summary", {})
    if summary:
        lines.append(f"**Averages:** mid-term bounce = {summary.get('midterm_avg_bounce', '?')} pts "
                      f"(n={summary.get('midterm_n', '?')}), election transition = "
                      f"{summary.get('election_avg_bounce', '?')} pts (n={summary.get('election_n', '?')})")
        lines.append("")

    lines.append("**Caveats:** Small n (9 events). Bounces conflated with contemporaneous events.")
    lines.append("")


def _write_crisis_section(lines, r):
    lines.append("### Rule 5: Crisis Rally Effect")
    lines.append("")
    lines.append("**Statement:** Crises tend to boost incumbent support, at least in the short term.")
    lines.append("")

    events = r.get("events", [])
    if events:
        lines.append("| Crisis | Incumbent | Before | +30d | +60d | +90d | +180d |")
        lines.append("|--------|-----------|--------|------|------|------|-------|")
        for e in events:
            inc_b = e.get("incumbent_before", "—")
            cols = []
            for d in [30, 60, 90, 180]:
                val = e.get(f"incumbent_change_{d}d")
                cols.append(f"{val:+.1f}" if val is not None else "—")
            lines.append(f"| {e['event'][:30]} | {e['incumbent']} | {inc_b} | {' | '.join(cols)} |")
        lines.append("")

    summary = r.get("summary", {})
    if summary:
        lines.append(f"**Average 30-day incumbent boost:** {summary.get('mean_incumbent_boost_30d', '?')} pts "
                      f"(n={summary.get('n_crises', '?')})")
        lines.append("")

    lines.append("**Caveats:** Only 4 crises; each is unique. COVID lockdown confounds rally with")
    lines.append("extensive policy response.")
    lines.append("")


def _write_economic_section(lines, r):
    lines.append("### Rule 6: Economic Sentiment")
    lines.append("")

    if r.get("error"):
        lines.append(f"*{r['error']}*")
        lines.append("")
        return

    n_polls = r.get("n_polls", 0)
    lines.append(f"**Statement:** Testing whether economic conditions predict incumbent support "
                  f"at poll-level (n={n_polls}) with various time lags.")
    lines.append("")

    lag_analysis = r.get("lag_analysis", {})
    if lag_analysis:
        lines.append("**Level effects by lag:**")
        lines.append("")
        lines.append("| Lag | Variable | Coefficient | p-value | Significant |")
        lines.append("|-----|----------|-------------|---------|-------------|")
        for lag_key in sorted(lag_analysis.keys()):
            for var, vdata in lag_analysis[lag_key].items():
                sig = "Yes" if vdata["significant"] else "No"
                lines.append(f"| {lag_key} | {var} | {vdata['coefficient']:.3f} | {vdata['p_value']:.4f} | {sig} |")
        lines.append("")

    change = r.get("change_analysis", {})
    if change:
        lines.append("**12-month change effects:**")
        lines.append("")
        lines.append("| Variable | Coefficient | p-value | Significant |")
        lines.append("|----------|-------------|---------|-------------|")
        for var, vdata in change.items():
            sig = "Yes" if vdata["significant"] else "No"
            lines.append(f"| {var} 12m Δ | {vdata['coefficient']:.3f} | {vdata['p_value']:.4f} | {sig} |")
        lines.append("")

    any_sig = any(
        vdata.get("significant")
        for lag_data in lag_analysis.values()
        for vdata in lag_data.values()
    ) or any(
        vdata.get("significant")
        for vdata in change.values()
    )

    if not any_sig:
        lines.append("**Finding:** Even with ~1,000 poll-level observations and monthly-interpolated")
        lines.append("economic data, no economic variable significantly predicts incumbent support.")
        lines.append("NZ polling appears remarkably insensitive to economic conditions — or annual")
        lines.append("economic data is too coarse to capture the relationship.")
    lines.append("")
    lines.append("**Caveats:** Economic data is interpolated from annual observations; true monthly/quarterly")
    lines.append("data might reveal relationships. Government fixed effects absorb between-government variation.")
    lines.append("")


def _write_convergence_section(lines, r):
    lines.append("### Rule 7: Election Proximity Behaviour")
    lines.append("")
    lines.append("**Statement:** Several consistent patterns emerge as elections approach: polling")
    lines.append("frequency surges, Labour's numbers settle, incumbents fade, and NZ First picks up")
    lines.append("late-deciding voters. But the National-Labour gap does not systematically narrow.")
    lines.append("")

    # Polling frequency
    freq = r.get("polling_frequency", {})
    if freq:
        lines.append("#### Polling Frequency")
        lines.append("")
        lines.append("| Days Out | Polls | Polls/Month |")
        lines.append("|----------|-------|-------------|")
        for lbl in ["365+", "180-365", "90-180", "30-90", "0-30"]:
            if lbl in freq:
                d = freq[lbl]
                lines.append(f"| {lbl} | {d['n']} | {d['polls_per_month']} |")
        lines.append("")

    # Poll-to-poll volatility
    vol = r.get("poll_volatility", {})
    if vol:
        lines.append("#### Poll-to-Poll Volatility")
        lines.append("")
        for party in ["National", "Labour"]:
            party_vol = vol.get(party, {})
            settle = vol.get(f"{party}_settling", {})
            if party_vol:
                lines.append(f"**{party}** — mean |change| by proximity:")
                lines.append("")
                lines.append("| Days Out | Mean |Δ| | Median |Δ| | N |")
                lines.append("|----------|---------|-----------|---|")
                for lbl in ["365+", "180-365", "90-180", "30-90", "0-30"]:
                    if lbl in party_vol:
                        d = party_vol[lbl]
                        lines.append(f"| {lbl} | {d['mean_abs_change']} | {d['median_abs_change']} | {d['n']} |")
                lines.append("")
            if settle:
                settles_str = "settles significantly" if settle.get("settles") else "no significant settling"
                lines.append(f"Late vs early: {settle.get('late_mean', '?')} vs {settle.get('early_mean', '?')} "
                              f"(p = {settle.get('p_value', '?')}) — {settles_str}")
                lines.append("")

    # Incumbent late fade
    inc_fade = r.get("incumbent_late_fade", {})
    inc_sum = r.get("incumbent_fade_summary", {})
    if inc_fade:
        lines.append("#### Incumbent Late Fade")
        lines.append("")
        if inc_sum:
            lines.append(f"Incumbents lose support in final 90 days in **{inc_sum['n_fades']}/{inc_sum['n_cycles']}** "
                          f"elections — campaign scrutiny erodes rather than builds incumbent position.")
            lines.append("")
        lines.append("| Election | Incumbent | 90-180d | 0-90d | Change |")
        lines.append("|----------|-----------|---------|-------|--------|")
        for year in sorted(inc_fade.keys()):
            d = inc_fade[year]
            lines.append(f"| {year} | {d['incumbent']} | {d['far_mean']} | {d['near_mean']} | {d['change']:+.1f} |")
        lines.append("")

    # National-Labour gap
    gap_cycles = r.get("gap_by_cycle", {})
    gap_sum = r.get("gap_summary", {})
    if gap_cycles:
        lines.append("#### National-Labour Gap (No Consistent Convergence)")
        lines.append("")
        if gap_sum:
            lines.append(f"Gap narrows in **{gap_sum['n_narrows']}/{gap_sum['n_cycles']}** cycles — ")
            lines.append("no systematic convergence; behaviour is election-specific.")
            lines.append("")
        lines.append("| Election | Early Gap | Late Gap | Change |")
        lines.append("|----------|-----------|----------|--------|")
        for year in sorted(gap_cycles.keys()):
            d = gap_cycles[year]
            direction = "narrows" if d["narrows"] else "widens"
            lines.append(f"| {year} | {d['early_gap']} | {d['late_gap']} | {d['change']:+.1f} ({direction}) |")
        lines.append("")

    # Minor party late surge
    minor_prox = r.get("minor_party_proximity", {})
    if minor_prox:
        lines.append("#### Minor Party Late Movement")
        lines.append("")
        lines.append("| Party | >180d Mean | <30d Mean | Change | p-value |")
        lines.append("|-------|-----------|-----------|--------|---------|")
        for party in ["Green", "ACT", "NZ First"]:
            if party in minor_prox:
                d = minor_prox[party]
                sig = " **" if d["significant"] else ""
                lines.append(f"| {party} | {d['early_mean']} | {d['late_mean']} | {d['change']:+.1f}{sig} | {d['p_value']} |")
        lines.append("")
        nzf = minor_prox.get("NZ First", {})
        if nzf and nzf.get("significant") and nzf.get("change", 0) > 0:
            lines.append(f"NZ First's late surge (+{nzf['change']} pts, p = {nzf['p_value']}) is consistent with their")
            lines.append("role as the party of late-deciding voters who are dissatisfied with both major parties.")
            lines.append("")

    # Total share
    total = r.get("total_share_by_proximity", {})
    if total:
        lines.append("#### Total Five-Party Share")
        lines.append("")
        vals = list(total.values())
        lines.append(f"Stable at {min(vals)}-{max(vals)}% regardless of proximity — the ~7% allocated to other")
        lines.append("parties and undecideds does not systematically shrink as elections approach.")
        lines.append("")

    lines.append("**Caveats:** Polling frequency increase means late-campaign bins have more data,")
    lines.append("potentially reducing noise. Incumbent fade conflated with natural fatigue trend.")
    lines.append("")


def _write_minor_section(lines, r):
    lines.append("### Rule 8: Minor Party Lifecycle Patterns")
    lines.append("")
    lines.append("**Statement:** Minor parties are systematically more volatile than major parties")
    lines.append("and their combined support acts as a dissatisfaction barometer.")
    lines.append("")

    cv = r.get("coefficient_of_variation", {})
    if cv:
        lines.append("**Coefficient of variation (std/mean):**")
        lines.append("")
        lines.append("| Party | Mean % | Std % | CV |")
        lines.append("|-------|--------|-------|----|")
        for party in ["National", "Labour", "Green", "ACT", "NZ First"]:
            if party in cv:
                d = cv[party]
                lines.append(f"| {party} | {d['mean']:.1f} | {d['std']:.1f} | {d['cv']:.3f} |")
        lines.append("")

    cv_comp = r.get("cv_comparison", {})
    if cv_comp:
        lines.append(f"Major party avg CV: {cv_comp['major_avg_cv']:.3f}, "
                      f"Minor party avg CV: {cv_comp['minor_avg_cv']:.3f} — "
                      f"minor parties are **{cv_comp['minor_avg_cv']/cv_comp['major_avg_cv']:.1f}x** more volatile")
        lines.append("")

    mi = r.get("minor_vs_incumbent", {})
    if mi:
        direction = "counter-cyclical (rises when incumbent falls)" if mi["counter_cyclical"] else "not significantly counter-cyclical"
        lines.append(f"**Minor total vs incumbent support:** coefficient = {mi['coefficient']:.3f} "
                      f"(p = {mi['p_value']:.4f}) — {direction}")
        lines.append("")

    thresh = r.get("mmp_threshold", {})
    if thresh:
        lines.append("**MMP 5% threshold stability:**")
        lines.append("")
        lines.append("| Party | % polls above 5% | Threshold crossings | N polls |")
        lines.append("|-------|-------------------|--------------------:|---------|")
        for party in ["Green", "ACT", "NZ First"]:
            if party in thresh:
                d = thresh[party]
                lines.append(f"| {party} | {d['pct_above_5']:.0f}% | {d['n_crossings']} | {d['n_polls']} |")
        lines.append("")

    lines.append("")


def _write_reset_section(lines, r):
    lines.append("### Rule 9: Post-Election Reset")
    lines.append("")
    lines.append("**Statement:** After elections, winner and loser support shifts from final polling levels.")
    lines.append("")

    transitions = r.get("transitions", [])
    if transitions:
        lines.append("| Election | Winner | Winner Δ | Loser | Loser Δ |")
        lines.append("|----------|--------|----------|-------|---------|")
        for t in transitions:
            wc = t.get("winner_change", "—")
            wc_str = f"{wc:+.1f}" if isinstance(wc, (int, float)) else wc
            lc = t.get("loser_change", "—")
            lc_str = f"{lc:+.1f}" if isinstance(lc, (int, float)) else lc
            lines.append(f"| {t['election']} | {t['winner']} | {wc_str} | {t['loser']} | {lc_str} |")
        lines.append("")

    summary = r.get("summary", {})
    if summary:
        lines.append(f"**Average winner change:** {summary.get('avg_winner_change', '?'):+.1f} pts "
                      f"({summary.get('winner_boost_count', 0)} up, {summary.get('winner_drop_count', 0)} down)")
        if summary.get("avg_loser_change") is not None:
            lines.append(f"**Average loser change:** {summary['avg_loser_change']:+.1f} pts")
        lines.append("")

    lines.append("**Caveats:** Post-election polls may reflect actual result rather than independent")
    lines.append("opinion shift. Coalition negotiations (1996, 2017) create extended uncertainty.")
    lines.append("")


def _write_thermostatic_section(lines, r):
    lines.append("### Rule 10: Thermostatic Model")
    lines.append("")
    lines.append("**Statement:** Public opinion acts as a thermostat (Wlezien 1995) — when a party's")
    lines.append("support rises above its equilibrium, it subsequently corrects downward, and vice versa.")
    lines.append("This creates a negative feedback loop that stabilises the political system.")
    lines.append("")

    for party in ["National", "Labour"]:
        pdata = r.get(party, {})
        ec = pdata.get("error_correction", {})
        if ec:
            lines.append(f"**{party} — error-correction model:**")
            lines.append(f"- β = {ec['beta']:.4f} (p = {ec['p_value']:.4f}, n = {ec['n']})")
            lines.append(f"- {ec['interpretation']}")
            lines.append(f"- Thermostatic correction {'confirmed' if ec['thermostatic'] else 'not confirmed'}")
            lines.append("")

        asym = pdata.get("asymmetric_correction", {})
        if asym:
            lines.append(f"**{party} — asymmetric correction:**")
            if "above_mean_beta" in asym:
                lines.append(f"- Above mean: β = {asym['above_mean_beta']:.4f} (p = {asym['above_mean_p']:.4f}, n = {asym['above_mean_n']})")
            if "below_mean_beta" in asym:
                lines.append(f"- Below mean: β = {asym['below_mean_beta']:.4f} (p = {asym['below_mean_p']:.4f}, n = {asym['below_mean_n']})")
            if "faster_correction" in asym:
                lines.append(f"- Faster correction: {asym['faster_correction']}")
            lines.append("")

    cross = r.get("cross_party_thermostat", {})
    if cross:
        lines.append("**Cross-party thermostat (high incumbent → opposition gains):**")
        lines.append(f"- β = {cross['beta']:.4f} (p = {cross['p_value']:.4f}, n = {cross['n']})")
        lines.append(f"- {cross['interpretation']}")
        lines.append("")

    lines.append("**Caveats:** Error-correction within government periods may conflate thermostatic")
    lines.append("correction with mean reversion in polling noise. Causal direction ambiguous —")
    lines.append("negative feedback could reflect genuine public response or polling methodology.")
    lines.append("")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Laws of NZ Politics: Statistical Analysis")
    print("=" * 55)

    print("\nLoading data...")
    df = load_all_polling_data()
    print(f"  {len(df)} polls loaded, {df['date'].min().date()} to {df['date'].max().date()}")

    print("\nRunning analyses...")
    all_results = {}

    all_results["incumbent_fatigue"] = analyze_incumbent_fatigue(df)
    all_results["punctuated_equilibrium"] = analyze_punctuated_equilibrium(df)
    all_results["cross_bloc"] = analyze_cross_bloc(df)
    all_results["leadership_honeymoon"] = analyze_leadership_honeymoon(df)
    all_results["crisis_rally"] = analyze_crisis_rally(df)
    all_results["economic_sentiment"] = analyze_economic_sentiment(df)
    all_results["election_convergence"] = analyze_election_convergence(df)
    all_results["minor_party_lifecycle"] = analyze_minor_party_lifecycle(df)
    all_results["post_election_reset"] = analyze_post_election_reset(df)
    all_results["thermostatic"] = analyze_thermostatic(df)

    print("\nGenerating report...")
    REPORTS_DIR.mkdir(exist_ok=True)
    report = generate_report(all_results)
    report_path = REPORTS_DIR / "nz_rules.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for s in _build_summaries(all_results):
        print(f"  {s['num']}. {s['name']}: {s['statement']} [{s['strength']}]")

    print(f"\nDone. Full report: {report_path}")


if __name__ == "__main__":
    main()
