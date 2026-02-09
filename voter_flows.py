#!/usr/bin/env python3
"""
Voter Flow Analysis: Partial Correlations & Compositional Data Analysis

Addresses the methodological limitation that simple pairwise correlations on
compositional data (vote shares summing to ~100%) produce spurious results.

Three approaches:
1. Partial correlations - correlate changes in Party A with Party B while
   controlling for all other parties
2. Log-return correlations - correlate proportional changes (Δln(xi)),
   which are not subject to the constant-sum constraint
3. Variation matrix (Aitchison, 1986) - var(ln(xi/xj)) for each pair,
   the standard CoDA measure of pairwise association
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import load_all_polling_data

PARTIES = ["National", "Labour", "Green", "ACT", "NZ First"]
REPORTS_DIR = Path(__file__).parent / "reports"

# Political bloc classification for interpreting correlations.
# Only within-bloc negative correlations indicate direct voter exchange;
# cross-bloc negatives reflect opposing political tides, not voter flow.
LEFT_BLOC = {"Labour", "Green"}
RIGHT_BLOC = {"National", "ACT", "NZ First"}


def same_bloc(p1: str, p2: str) -> bool:
    return ({p1, p2} <= LEFT_BLOC) or ({p1, p2} <= RIGHT_BLOC)


def prepare_data(df: pd.DataFrame):
    """
    Filter to polls with all 5 major parties and compute within-cycle
    changes in both raw percentage points and log-returns.

    Returns (levels_df, raw_changes_df, logret_changes_df).
    """
    df_valid = df.dropna(subset=PARTIES).copy()
    df_valid = df_valid.sort_values("date").reset_index(drop=True)

    # Replace zeros before taking logs
    for p in PARTIES:
        df_valid[p] = df_valid[p].clip(lower=0.01)

    raw_changes = []
    logret_changes = []

    for year, group in df_valid.groupby("election_year"):
        group = group.sort_values("date")

        # Raw changes (percentage point diffs)
        rdiff = group[PARTIES].diff().iloc[1:]
        rdiff["election_year"] = year
        rdiff["date"] = group["date"].iloc[1:].values
        raw_changes.append(rdiff)

        # Log-return changes: Δln(xi) = ln(xi_t / xi_{t-1})
        log_vals = np.log(group[PARTIES])
        ldiff = log_vals.diff().iloc[1:]
        ldiff["election_year"] = year
        ldiff["date"] = group["date"].iloc[1:].values
        logret_changes.append(ldiff)

    raw_df = pd.concat(raw_changes, ignore_index=True) if raw_changes else pd.DataFrame()
    logret_df = pd.concat(logret_changes, ignore_index=True) if logret_changes else pd.DataFrame()

    return df_valid, raw_df, logret_df


def partial_correlation_matrix(change_df: pd.DataFrame):
    """
    Compute partial correlation matrix from the 5-party change vectors.

    pcorr(i,j) = -P[i,j] / sqrt(P[i,i] * P[j,j])
    where P is the precision matrix (inverse of covariance matrix).
    """
    data = change_df[PARTIES].dropna()
    n = len(data)
    k = len(PARTIES)

    corr_matrix = data.corr().values

    try:
        precision = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(corr_matrix)

    pcorr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                pcorr[i, j] = 1.0
            else:
                pcorr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])

    # P-values: df = n - 2 - (k - 2) controlling variables
    dof = n - 2 - (k - 2)
    pvals = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                pvals[i, j] = 0.0
            else:
                r = pcorr[i, j]
                if abs(r) >= 1.0:
                    pvals[i, j] = 0.0
                else:
                    t_stat = r * np.sqrt(dof / (1 - r**2))
                    pvals[i, j] = 2 * stats.t.sf(abs(t_stat), dof)

    pcorr_df = pd.DataFrame(pcorr, index=PARTIES, columns=PARTIES)
    pval_df = pd.DataFrame(pvals, index=PARTIES, columns=PARTIES)
    return pcorr_df, pval_df, n


def logreturn_correlation_analysis(logret_df: pd.DataFrame):
    """
    Correlation matrix of log-returns Δln(xi).

    Log-returns are NOT subject to the constant-sum constraint
    (Δln(x1) + ... + Δln(x5) ≠ constant), so their correlations
    reflect genuine co-movement without compositional bias.
    """
    data = logret_df[PARTIES].dropna()
    n = len(data)

    corr_matrix = data.corr()

    pvals = pd.DataFrame(np.ones((len(PARTIES), len(PARTIES))),
                         index=PARTIES, columns=PARTIES)
    for i, p1 in enumerate(PARTIES):
        for j, p2 in enumerate(PARTIES):
            if i != j:
                _, p = stats.pearsonr(data[p1], data[p2])
                pvals.loc[p1, p2] = p
            else:
                pvals.loc[p1, p2] = 0.0

    return corr_matrix, pvals, n


def variation_matrix(levels_df: pd.DataFrame):
    """
    Aitchison's variation matrix: τ(i,j) = var(ln(xi/xj)).

    Computed on within-cycle consecutive poll log-ratios.
    Low τ = parties maintain stable ratio (co-movement / proportionality).
    High τ = ratio fluctuates (potential voter exchange or independent shocks).
    """
    k = len(PARTIES)
    tau = np.zeros((k, k))

    # Compute log-ratios for each pair, then within-cycle changes
    for ii, p1 in enumerate(PARTIES):
        for jj, p2 in enumerate(PARTIES):
            if ii == jj:
                continue
            # ln(xi/xj) for each poll
            lr = np.log(levels_df[p1].clip(lower=0.01) / levels_df[p2].clip(lower=0.01))

            # Within-cycle changes of the log-ratio
            lr_changes = []
            for _, group_idx in levels_df.groupby("election_year").groups.items():
                group_lr = lr.loc[group_idx]
                diff = group_lr.diff().iloc[1:]
                lr_changes.append(diff)

            if lr_changes:
                all_changes = pd.concat(lr_changes).dropna()
                tau[ii, jj] = all_changes.var()

    tau_df = pd.DataFrame(tau, index=PARTIES, columns=PARTIES)
    return tau_df


def find_flow_corridors(logret_corr: pd.DataFrame, logret_pval: pd.DataFrame,
                        pcorr_df: pd.DataFrame, pcorr_pval: pd.DataFrame,
                        tau_df: pd.DataFrame):
    """Rank party pairs by log-return correlation with bloc classification."""
    corridors = []

    for i, p1 in enumerate(PARTIES):
        for j, p2 in enumerate(PARTIES):
            if j <= i:
                continue

            corridors.append({
                "pair": f"{p1} <-> {p2}",
                "p1": p1,
                "p2": p2,
                "lr_r": logret_corr.loc[p1, p2],
                "lr_p": logret_pval.loc[p1, p2],
                "partial_r": pcorr_df.loc[p1, p2],
                "partial_p": pcorr_pval.loc[p1, p2],
                "tau": tau_df.loc[p1, p2],
                "within_bloc": same_bloc(p1, p2),
            })

    corridors.sort(key=lambda x: x["lr_r"])
    return corridors


def format_matrix(df: pd.DataFrame, pval_df: pd.DataFrame = None,
                  fmt: str = ".3f") -> str:
    """Format a matrix for display, with optional significance stars."""
    lines = []

    header = f"{'':>12}" + "".join(f"{p:>12}" for p in df.columns)
    lines.append(header)
    lines.append("-" * len(header))

    for row in df.index:
        parts = [f"{row:>12}"]
        for col in df.columns:
            val = df.loc[row, col]
            if pval_df is not None and row != col:
                p = pval_df.loc[row, col]
                stars = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            else:
                stars = "   "
            parts.append(f"{val:>8{fmt}}{stars}")
        lines.append("".join(parts))

    if pval_df is not None:
        lines.append("")
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    return "\n".join(lines)


def _interpret_pair(c: dict) -> str:
    """Generate a short interpretation label for a party pair."""
    if c["lr_p"] >= 0.05:
        return "Not significant"
    if c["lr_r"] > 0.05:
        return "Minor party co-movement"
    if c["lr_r"] > -0.05:
        return "Negligible"
    # Negative and significant
    if c["within_bloc"]:
        return "Within-bloc voter exchange"
    if c["pair"] == "National <-> Labour":
        return "Swing voter corridor"
    return "Cross-bloc opposing tides"


def generate_report(raw_corr, raw_n,
                    pcorr_df, pcorr_pval, pcorr_n,
                    logret_corr, logret_pval, logret_n,
                    tau_df, corridors) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Voter Flow Analysis: Partial Correlations & Compositional Data")
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append("Simple pairwise correlations on vote share data are misleading because")
    lines.append("party shares are **compositional** (they sum to ~100%). A decline in one")
    lines.append("party mechanically inflates others, creating spurious negative correlations.")
    lines.append("Three methods address this:")
    lines.append("")
    lines.append("1. **Partial correlations** - control for all other parties")
    lines.append("2. **Log-return correlations** - correlate proportional changes Δln(xi),")
    lines.append("   which are free from the constant-sum constraint")
    lines.append("3. **Variation matrix** (Aitchison, 1986) - var(ln(xi/xj)) measures how")
    lines.append("   much the ratio between two parties fluctuates")
    lines.append("")

    # 1. Raw correlations
    lines.append("## 1. Raw Change Correlations (for reference)")
    lines.append("")
    lines.append(f"Poll-to-poll changes in raw percentage points (n = {raw_n}).")
    lines.append("These are **biased** by the compositional constraint.")
    lines.append("")
    lines.append("```")
    lines.append(format_matrix(raw_corr))
    lines.append("```")
    lines.append("")

    # 2. Partial correlations
    lines.append("## 2. Partial Correlations")
    lines.append("")
    lines.append(f"Each correlation controls for changes in all other parties (n = {pcorr_n}).")
    lines.append("")
    lines.append("```")
    lines.append(format_matrix(pcorr_df, pcorr_pval))
    lines.append("```")
    lines.append("")
    lines.append("**Caveat:** All partial correlations are strongly negative. With 5 party")
    lines.append("changes that approximately sum to zero, conditioning on the other 3 parties")
    lines.append("nearly determines the sum of the remaining 2, forcing a negative correlation.")
    lines.append("The relative ordering is informative (National-Labour is the most negative,")
    lines.append("ACT-NZ First the least), but absolute values are artifacts of the closure.")
    lines.append("")

    # 3. Log-return correlations (primary method)
    lines.append("## 3. Log-Return Correlations (primary method)")
    lines.append("")
    lines.append(f"Correlations of Δln(xi) — proportional changes (n = {logret_n}).")
    lines.append("Unlike raw changes, log-returns **do not** sum to a constant, so these")
    lines.append("correlations are free from compositional bias.")
    lines.append("")
    lines.append("Interpretation requires political context (Left bloc: Labour, Green;")
    lines.append("Right bloc: National, ACT, NZ First):")
    lines.append("- **Within-bloc negative** = direct voter exchange (e.g. Labour ↔ Green)")
    lines.append("- **Cross-bloc negative** = opposing tides, not direct flow (e.g. National ↔ Green)")
    lines.append("- **Positive** = co-movement from shared external drivers")
    lines.append("")
    lines.append("```")
    lines.append(format_matrix(logret_corr, logret_pval))
    lines.append("```")
    lines.append("")

    # 4. Variation matrix
    lines.append("## 4. Variation Matrix — var(ln(xi/xj))")
    lines.append("")
    lines.append("Aitchison's standard measure for compositional data. Each cell shows")
    lines.append("how much the log-ratio between two parties fluctuates between consecutive")
    lines.append("polls. **Low** = stable ratio (co-movement); **High** = volatile ratio")
    lines.append("(potential voter exchange or independent shocks).")
    lines.append("")
    lines.append("```")
    lines.append(format_matrix(tau_df, fmt=".4f"))
    lines.append("```")
    lines.append("")

    # 5. Flow corridors
    lines.append("## 5. Pairwise Relationships (ranked by log-return correlation)")
    lines.append("")
    lines.append("Bloc key: L = Left (Labour, Green), R = Right (National, ACT, NZ First)")
    lines.append("")
    lines.append("| Rank | Party Pair | Bloc | Log-ret r | p-value | Variation | Interpretation |")
    lines.append("|------|-----------|------|-----------|---------|-----------|----------------|")

    for i, c in enumerate(corridors, 1):
        lr_sig = "***" if c["lr_p"] < 0.001 else "**" if c["lr_p"] < 0.01 else "*" if c["lr_p"] < 0.05 else ""
        bloc = "within" if c["within_bloc"] else "cross"

        interp = _interpret_pair(c)

        lines.append(
            f"| {i} | {c['pair']} | {bloc} | {c['lr_r']:.3f}{lr_sig} | {c['lr_p']:.4f} "
            f"| {c['tau']:.4f} | {interp} |"
        )

    lines.append("")

    # 6. Key findings — grouped by type
    lines.append("## 6. Key Findings")
    lines.append("")

    within_exchange = [c for c in corridors
                       if c["within_bloc"] and c["lr_r"] < -0.05 and c["lr_p"] < 0.05]
    cross_oppose = [c for c in corridors
                    if not c["within_bloc"] and c["lr_r"] < -0.05 and c["lr_p"] < 0.05]
    sig_comove = [c for c in corridors
                  if c["lr_r"] > 0.05 and c["lr_p"] < 0.05]
    # Special: National-Labour is cross-bloc but the main swing voter corridor
    nat_lab = [c for c in corridors if c["pair"] == "National <-> Labour"]

    if within_exchange:
        lines.append("### Within-bloc voter exchange (direct competition for same voters):")
        lines.append("")
        for c in within_exchange:
            p1, p2 = c["p1"], c["p2"]
            bloc_name = "left" if {p1, p2} <= LEFT_BLOC else "right"
            lines.append(f"- **{c['pair']}**: r = {c['lr_r']:.3f} (p = {c['lr_p']:.4f}), "
                         f"τ = {c['tau']:.4f}")
            lines.append(f"  Direct competition within the {bloc_name} bloc. When {p1}")
            lines.append(f"  gains, {p2} tends to lose — consistent with voters shifting")
            lines.append(f"  between ideologically adjacent parties.")
        lines.append("")

    if nat_lab and nat_lab[0]["lr_p"] < 0.05 and nat_lab[0]["lr_r"] < 0:
        c = nat_lab[0]
        lines.append("### Swing voter corridor:")
        lines.append("")
        lines.append(f"- **National <-> Labour**: r = {c['lr_r']:.3f} (p = {c['lr_p']:.4f}), "
                     f"τ = {c['tau']:.4f}")
        lines.append(f"  The two major parties compete for centrist swing voters.")
        lines.append(f"  The very low variation (τ = {c['tau']:.4f}) indicates their")
        lines.append(f"  combined share is highly stable — the overall left-right")
        lines.append(f"  balance shifts slowly.")
        lines.append("")

    cross_oppose_no_natlab = [c for c in cross_oppose
                              if c["pair"] != "National <-> Labour"]
    if cross_oppose_no_natlab:
        lines.append("### Cross-bloc opposing tides (not direct voter exchange):")
        lines.append("")
        lines.append("These pairs move inversely because they sit on opposite sides")
        lines.append("of the left-right divide. When the political environment favours")
        lines.append("one bloc, the other loses — but voters typically flow through")
        lines.append("intermediaries (e.g., Green → Labour → swing → National), not")
        lines.append("directly between ideologically distant parties.")
        lines.append("")
        for c in cross_oppose_no_natlab:
            lines.append(f"- **{c['pair']}**: r = {c['lr_r']:.3f} (p = {c['lr_p']:.4f})")
        lines.append("")

    if sig_comove:
        lines.append("### Minor party co-movement:")
        lines.append("")
        lines.append("These pairs rise and fall together. When major parties dominate")
        lines.append("polling, all minor parties tend to be squeezed simultaneously;")
        lines.append("when dissatisfaction with major parties rises, minor parties")
        lines.append("benefit collectively.")
        lines.append("")
        for c in sig_comove:
            lines.append(f"- **{c['pair']}**: r = {c['lr_r']:.3f} (p = {c['lr_p']:.4f})")
        lines.append("")

    # Methodological notes
    lines.append("## Methodological Notes")
    lines.append("")
    lines.append("- **Partial correlations** use the precision matrix (inverse of the")
    lines.append("  correlation matrix). With near-compositional data, all partial")
    lines.append("  correlations tend negative (Aitchison, 1986). Included for")
    lines.append("  completeness; log-return correlations are the primary measure.")
    lines.append("- **Log-return correlations** correlate Δln(xi) between consecutive")
    lines.append("  polls. Since log-returns do not sum to a constant, these are free")
    lines.append("  from compositional bias. This is analogous to correlating asset")
    lines.append("  returns in finance — a well-established approach for proportional data.")
    lines.append("- **Variation matrix** entries τ(i,j) = var(Δln(xi/xj)). This is")
    lines.append("  Aitchison's standard pairwise association measure for compositional")
    lines.append("  data and does not suffer from CLR singularity issues.")
    lines.append("- Changes are computed **within election cycles only** to avoid")
    lines.append("  cross-election discontinuities.")
    lines.append("- Only polls reporting all 5 major parties are included.")
    lines.append("- Shares of 0% are replaced with 0.01% before taking logs.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by voter_flows.py*")

    return "\n".join(lines)


def main():
    print("Voter Flow Analysis")
    print("=" * 60)

    # Load and prepare data
    print("\nLoading polling data...")
    df = load_all_polling_data()
    n_total = len(df)
    n_valid = df.dropna(subset=PARTIES).shape[0]
    print(f"  Total polls: {n_total}")
    print(f"  Polls with all 5 parties: {n_valid}")

    print("\nComputing within-cycle changes...")
    levels_df, raw_change_df, logret_df = prepare_data(df)
    n_changes = len(raw_change_df)
    print(f"  Change observations: {n_changes}")

    if n_changes < 20:
        print("ERROR: Too few observations for meaningful analysis.")
        return

    # 1. Raw correlations
    print("\nRaw change correlations (biased by composition)...")
    raw_data = raw_change_df[PARTIES].dropna()
    raw_corr = raw_data.corr()
    raw_n = len(raw_data)
    print(format_matrix(raw_corr))

    # 2. Partial correlations
    print("\n" + "=" * 60)
    print("PARTIAL CORRELATIONS (controlling for other parties)")
    print("=" * 60)
    pcorr_df, pcorr_pval, pcorr_n = partial_correlation_matrix(raw_change_df)
    print(format_matrix(pcorr_df, pcorr_pval))

    # 3. Log-return correlations
    print("\n" + "=" * 60)
    print("LOG-RETURN CORRELATIONS (compositionally unbiased)")
    print("=" * 60)
    logret_corr, logret_pval, logret_n = logreturn_correlation_analysis(logret_df)
    print(format_matrix(logret_corr, logret_pval))

    # 4. Variation matrix
    print("\n" + "=" * 60)
    print("VARIATION MATRIX — var(Δln(xi/xj))")
    print("=" * 60)
    tau_df = variation_matrix(levels_df)
    print(format_matrix(tau_df, fmt=".4f"))

    # 5. Flow corridors
    print("\n" + "=" * 60)
    print("VOTER FLOW CORRIDORS (ranked by log-return correlation)")
    print("=" * 60)
    corridors = find_flow_corridors(logret_corr, logret_pval, pcorr_df, pcorr_pval, tau_df)
    for i, c in enumerate(corridors, 1):
        sig = "*" if c["lr_p"] < 0.05 else " "
        bloc = "within" if c["within_bloc"] else "cross"
        print(f"  {i}. {c['pair']:>25}  r = {c['lr_r']:+.3f} {sig}  "
              f"τ = {c['tau']:.4f}  [{bloc:>6}]  {_interpret_pair(c)}")

    # Generate and save report
    REPORTS_DIR.mkdir(exist_ok=True)
    report = generate_report(raw_corr, raw_n,
                             pcorr_df, pcorr_pval, pcorr_n,
                             logret_corr, logret_pval, logret_n,
                             tau_df, corridors)
    report_path = REPORTS_DIR / "voter_flows.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
