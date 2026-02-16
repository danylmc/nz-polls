#!/usr/bin/env python3
"""
Phase 1: Data Infrastructure for Deep NZ Politics Analysis

Builds three foundational datasets:
1. Quarterly economic time series from Stats NZ CSVs (GDP + CPI)
2. Poll-economy merged dataset (each poll linked to economic conditions)
3. Harmonized NZES cross-election dataset (individual-level survey data)
"""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
GDP_CSV = DATA_DIR / "gross-domestic-product-september-2025-quarter.csv"
CPI_CSV = DATA_DIR / "consumers-price-index-december-2025-quarter-index-numbers (1).csv"
NZES_DB = DATA_DIR / "nzesdb.sqlite"

OUTPUT_ECON = DATA_DIR / "quarterly_economics.csv"
OUTPUT_POLLS = DATA_DIR / "polls_with_economics.csv"
OUTPUT_NZES = DATA_DIR / "nzes_harmonized.csv"

# ─── 1a. Quarterly Economic Time Series ─────────────────────────────────────────

GDP_SERIES = {
    "gdp_real": "SNEQ.SG02RSC00B15",          # Real GDP (chain volume, SA)
    "gdp_growth_qoq": "SNEQ.SG02RSC31B15PC",  # GDP growth % q-on-q (SA)
    "gdp_per_capita": "SNEQ.SG09RSC00B15NZ",   # Real GDP per capita (chain volume, SA)
}

CPI_SERIES = {
    "cpi_all": "CPIQ.SE9A",       # All Groups CPI index
    "cpi_food": "CPIQ.SE901",     # Food
    "cpi_housing": "CPIQ.SE904",  # Housing and household utilities
    "cpi_petrol": "CPIQ.SE907202",  # Petrol
}


def parse_stats_nz_period(period_str):
    """Convert Stats NZ period format '2020.03' to a datetime (end of quarter)."""
    year, month = str(period_str).split(".")
    year, month = int(year), int(month)
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def extract_quarterly_economics():
    """Extract quarterly GDP and CPI data from Stats NZ CSVs."""
    print("Extracting quarterly economic data...")

    # --- GDP ---
    gdp_raw = pd.read_csv(GDP_CSV)
    gdp_frames = {}
    for col_name, series_ref in GDP_SERIES.items():
        subset = gdp_raw[gdp_raw["Series_reference"] == series_ref].copy()
        subset = subset[["Period", "Data_value"]].dropna(subset=["Data_value"])
        subset["quarter"] = subset["Period"].apply(parse_stats_nz_period)
        subset = subset.rename(columns={"Data_value": col_name})
        gdp_frames[col_name] = subset[["quarter", col_name]].set_index("quarter")
        print(f"  {col_name} ({series_ref}): {len(subset)} quarters")

    # --- CPI ---
    cpi_raw = pd.read_csv(CPI_CSV)
    cpi_frames = {}
    for col_name, series_ref in CPI_SERIES.items():
        subset = cpi_raw[cpi_raw["Series_reference"] == series_ref].copy()
        subset = subset[["Period", "Data_value"]].copy()
        subset["Data_value"] = pd.to_numeric(subset["Data_value"], errors="coerce")
        subset = subset[subset["Data_value"] > 0].dropna(subset=["Data_value"])
        subset["quarter"] = subset["Period"].apply(parse_stats_nz_period)
        subset = subset.rename(columns={"Data_value": col_name})
        cpi_frames[col_name] = subset[["quarter", col_name]].set_index("quarter")
        print(f"  {col_name} ({series_ref}): {len(subset)} quarters")

    # Merge all series
    merged = pd.DataFrame(index=pd.DatetimeIndex([], name="quarter"))
    for name, frame in {**gdp_frames, **cpi_frames}.items():
        merged = merged.join(frame, how="outer")
    merged = merged.sort_index()

    # Compute year-on-year GDP growth from levels
    if "gdp_real" in merged.columns:
        merged["gdp_growth_yoy"] = merged["gdp_real"].pct_change(4) * 100

    # Compute inflation rates (year-on-year % change from index levels)
    for cpi_col, inflation_col in [
        ("cpi_all", "inflation_yoy"),
        ("cpi_food", "food_inflation"),
        ("cpi_housing", "housing_inflation"),
        ("cpi_petrol", "petrol_inflation"),
    ]:
        if cpi_col in merged.columns:
            merged[inflation_col] = merged[cpi_col].pct_change(4) * 100

    # Filter to the range where we have meaningful data
    merged = merged[merged.index >= "1988-01-01"]

    merged.to_csv(OUTPUT_ECON)
    print(f"\nSaved quarterly economics: {OUTPUT_ECON}")
    print(f"  Shape: {merged.shape}")
    print(f"  Date range: {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"  Columns: {list(merged.columns)}")

    return merged


# ─── 1b. Poll-Economy Merged Dataset ────────────────────────────────────────────

GOVERNMENT_TIMELINE = [
    ("1990-11-02", "1996-12-15", "National"),
    ("1996-12-16", "1999-12-05", "National"),
    ("1999-12-06", "2008-11-19", "Labour"),
    ("2008-11-19", "2017-10-26", "National"),
    ("2017-10-26", "2023-11-27", "Labour"),
    ("2023-11-27", "2030-01-01", "National"),
]


def get_incumbent_party(date):
    """Return the governing party for a given date."""
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
    for start, end, party in GOVERNMENT_TIMELINE:
        if start <= date_str <= end:
            return party
    return None


def load_all_polls():
    """Load all party vote polling data from JSON files."""
    polls = []
    for json_file in sorted(DATA_DIR.glob("*_polling.json")):
        if "_pm_" in json_file.name:
            continue
        with open(json_file) as f:
            data = json.load(f)
        election_year = data["election_year"]
        for poll in data["polls"]:
            if poll.get("date"):
                row = {
                    "date": poll["date"],
                    "election_year": election_year,
                    "pollster": poll.get("pollster"),
                    "sample_size": poll.get("sample_size"),
                }
                for party, pct in poll.get("parties", {}).items():
                    if pct is not None:
                        row[party] = pct
                polls.append(row)

    df = pd.DataFrame(polls)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_polls_with_economics(econ_df):
    """Merge each poll with the most recent completed economic quarter + lags."""
    print("\nMerging polls with economic data...")

    polls = load_all_polls()
    print(f"  Loaded {len(polls)} polls")

    polls["incumbent"] = polls["date"].apply(get_incumbent_party)

    def get_incumbent_support(row):
        inc = row.get("incumbent")
        if inc and inc in row.index:
            val = row.get(inc)
            if pd.notna(val):
                return val
        return np.nan

    polls["incumbent_support"] = polls.apply(get_incumbent_support, axis=1)

    econ_quarters = econ_df.index.sort_values()
    econ_cols = [
        "gdp_real", "gdp_growth_qoq", "gdp_growth_yoy", "gdp_per_capita",
        "inflation_yoy", "food_inflation", "housing_inflation", "petrol_inflation",
    ]
    available_econ_cols = [c for c in econ_cols if c in econ_df.columns]

    result_rows = []
    for _, poll in polls.iterrows():
        poll_date = poll["date"]
        prior_quarters = econ_quarters[econ_quarters <= poll_date]
        row = poll.to_dict()
        if len(prior_quarters) > 0:
            for lag in [0, 1, 2, 4]:
                idx = len(prior_quarters) - 1 - lag
                if idx >= 0:
                    q = prior_quarters[idx]
                    suffix = "" if lag == 0 else f"_lag{lag}"
                    for col in available_econ_cols:
                        val = econ_df.loc[q, col] if q in econ_df.index else np.nan
                        row[f"{col}{suffix}"] = val
        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT_POLLS, index=False)
    print(f"  Saved polls with economics: {OUTPUT_POLLS}")
    print(f"  Shape: {result.shape}")
    return result


# ─── 1c. NZES Harmonized Dataset ────────────────────────────────────────────────

PARTY_VOTE_RECODE = {
    1996: {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT",
           6: "Christian", 7: "United", 14: "Green"},
    1999: {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT",
           6: "Christian", 7: "Green", 10: "United"},
    2002: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           7: "Alliance", 9: "Progressive", 11: "United Future"},
    2005: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
    2008: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
    2011: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori"},
    2014: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 11: "Conservative"},
    2017: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "Maori", 8: "TOP"},
    2020: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "Maori", 10: "TOP"},
    2023: {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First",
           6: "Maori"},
}

PREV_VOTE_RECODE = {
    1999: {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 6: "ACT"},
    2002: {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT", 6: "Green"},
    2005: {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT", 6: "Green"},
    2008: {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT", 6: "Green"},
    2011: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
    2014: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
    2017: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
    2020: {2: "Labour", 3: "National", 4: "Green", 5: "NZ First", 6: "ACT"},
    2023: {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First", 6: "Maori"},
}

NZES_VAR_MAP = {
    1996: {
        "party_vote": "VOT96P",
        "lr_self": "CSCAL", "lr_national": "CSCNAT", "lr_labour": "CSCLAB",
        "econ_state": "ECSAT",
        "age": "QAGE", "sex": "QSEX",
        "ethnic_id": "QETHN",
    },
    1999: {
        "party_vote": "PVOTE", "prev_vote": "CVOT96P",
        "lr_self": "SCALE", "lr_national": "SCNAT", "lr_labour": "SCLAB",
        "econ_state": "CECON", "econ_change": "CECCH",
        "age": "AGE", "sex": "SEX",
        "education": "CEDQUL", "income": "SRINCUM",
        "ethnicity_euro": "CETHNEU", "ethnicity_maori": "CETHNMR",
    },
    2002: {
        "party_vote": "WVOT02P", "prev_vote": "WVOT99P",
        "lr_self": "WLRSCLE", "lr_national": "WNTSCL", "lr_labour": "WLBSCL",
        "econ_state": "WECSTAT",
        "age": "WNAGE", "sex": "WSEX",
        "ethnic_id": "WETHID",
    },
    2005: {
        "party_vote": "yvot05p", "prev_vote": "yvot02p",
        "lr_self": "ylscale", "lr_national": "ynatscl", "lr_labour": "ylabscl",
        "econ_state": "yecstat", "econ_change": "ycounow",
        "age": "YAGE", "sex": "ysex",
        "income": "ypincum",
        "ethnicity_euro": "yetheur", "ethnicity_maori": "yethmao",
    },
    2008: {
        "party_vote": "zvt08p", "prev_vote": "zvot05p",
        "lr_self": "zscale", "lr_national": "zsclnat", "lr_labour": "zscllab",
        "econ_change": "zretcon",
        "age": "zborn", "age_is_birthyear": True,
        "sex": "zsex",
        "education": "zeducat", "income": "zhincom",
    },
    2011: {
        "party_vote": "jpartyvote", "prev_vote": "jlastpvote",
        "lr_self": "jslflr", "lr_national": "jnatlr", "lr_labour": "jlablr",
        "lr_green": "jgrnlr",
        "econ_change": "jecostate",
        "age": "jage", "sex": "jsex",
        "education": "jhqual", "income": "jhhincome",
        "ethnicity_euro": "jethnicity_e", "ethnicity_maori": "jethnicity_m",
        "party_like_lab": "jlablike", "party_like_nat": "jnatlike",
        "party_like_grn": "jgrnlike", "party_like_nzf": "jnzflike",
        "party_like_act": "jactlike",
    },
    2014: {
        "party_vote": "dpartyvote", "prev_vote": "dlastpvote",
        "lr_self": "dslflr", "lr_national": "dnatlr", "lr_labour": "dlablr",
        "lr_green": "dgrnlr",
        "econ_change": "decostate",
        "age": "dage", "sex": "dsex",
        "education": "coldedcons", "income": "dhhincome",
        "ethnicity_euro": "dethnicity_e", "ethnicity_maori": "dethnicity_m",
        "party_like_lab": "dlablike", "party_like_nat": "dnatlike",
        "party_like_grn": "dgrnlike", "party_like_nzf": "dnzflike",
        "party_like_act": "dactlike",
    },
    2017: {
        "party_vote": "rpartyvote", "prev_vote": "rlastpvote",
        "lr_self": "rselflr", "lr_national": "rnationallr", "lr_labour": "rlabourlr",
        "lr_green": "rgreenlr",
        "econ_change": "recostate",
        "age": "rsamage", "sex": "rsex",
        "education": "reducate", "income": "rhhincome",
        "ethnicity_euro": "reuroeth", "ethnicity_maori": "rmaorieth",
        "party_like_lab": "rlablike", "party_like_nat": "rnatlike",
        "party_like_grn": "rgrnlike", "party_like_nzf": "rnzflike",
        "party_like_act": "ractlike",
    },
    2020: {
        "party_vote": "E3", "prev_vote": "E19",
        "lr_self": "B8", "lr_national": "B7_2", "lr_labour": "B7_1",
        "lr_green": "B7_3",
        "econ_change": "C17_2",
        "age": "lmage", "sex": "G1",
        "education": "G5", "income": "G33",
        "ethnicity_euro": "G28_1", "ethnicity_maori": "G28_2",
    },
    2023: {
        "party_vote": "mpartyvote", "prev_vote": "mppartyvote",
        "lr_national": "B5b", "lr_labour": "B5a", "lr_green": "B5c",
        "econ_change": "A14",
        "age": "mage", "sex": "H2",
        "education": "meducate", "income": "H11",
        "ethnicity_euro": "H24a", "ethnicity_maori": "H24b",
    },
}

SURVEY_ELECTION_YEARS = {
    "nzes_1996": 1996, "nzes_1999_post": 1999, "nzes_2002": 2002,
    "nzes_2005": 2005, "nzes_2008": 2008, "nzes_2011": 2011,
    "nzes_2014": 2014, "nzes_2017": 2017, "nzes_2020": 2020, "nzes_2023": 2023,
}
TABLE_FOR_YEAR = {v: k for k, v in SURVEY_ELECTION_YEARS.items()}

EDUCATION_RECODE = {
    2008: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3},
    2011: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3},
    2014: {0: 0, 1: 1, 2: 2, 3: 3},
    2017: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
           10: 2, 11: 2, 12: 3, 13: 3, 14: 3},
    # 2020 (G5): 1=None, 2-4=School L1-L3, 5-8=Diploma/L4-L7, 9=Bachelor, 10=Hons, 11=Masters, 12=PhD
    2020: {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3,
           11: 3, 12: 3},
    2023: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
}


def harmonize_nzes():
    """Build harmonized cross-election NZES dataset."""
    print("\nHarmonizing NZES data across elections...")

    conn = sqlite3.connect(str(NZES_DB))
    all_respondents = []

    for year, var_map in sorted(NZES_VAR_MAP.items()):
        table = TABLE_FOR_YEAR[year]
        print(f"\n  Processing {table} ({year})...")

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"    Raw rows: {len(df)}")

        respondents = []
        for _, raw_row in df.iterrows():
            row = {"election_year": year}

            # Party vote
            pv_var = var_map.get("party_vote")
            if pv_var and pv_var in raw_row.index:
                pv_val = raw_row[pv_var]
                if pd.notna(pv_val):
                    pv_int = int(pv_val)
                    recode = PARTY_VOTE_RECODE.get(year, {})
                    row["party_vote"] = recode.get(pv_int)
                    if pv_int == 0 or pv_int >= 77:
                        row["party_vote"] = None

            # Previous vote
            prev_var = var_map.get("prev_vote")
            if prev_var and prev_var in raw_row.index:
                prev_val = raw_row[prev_var]
                if pd.notna(prev_val):
                    prev_int = int(prev_val)
                    recode = PREV_VOTE_RECODE.get(year, {})
                    row["prev_vote"] = recode.get(prev_int)
                    if prev_int == 0 or prev_int >= 77:
                        row["prev_vote"] = None

            # Left-right self-placement (0-10 scale)
            lr_var = var_map.get("lr_self")
            if lr_var and lr_var in raw_row.index:
                lr_val = raw_row[lr_var]
                if pd.notna(lr_val):
                    lr_f = float(lr_val)
                    if 0 <= lr_f <= 10:
                        row["lr_self"] = lr_f

            # Left-right party placements
            for lr_key in ["lr_national", "lr_labour", "lr_green"]:
                lr_var = var_map.get(lr_key)
                if lr_var and lr_var in raw_row.index:
                    lr_val = raw_row[lr_var]
                    if pd.notna(lr_val):
                        lr_f = float(lr_val)
                        if 0 <= lr_f <= 10:
                            row[lr_key] = lr_f

            # Economic perceptions
            for econ_key in ["econ_state", "econ_change"]:
                econ_var = var_map.get(econ_key)
                if econ_var and econ_var in raw_row.index:
                    econ_val = raw_row[econ_var]
                    if pd.notna(econ_val):
                        ev = float(econ_val)
                        if 1 <= ev <= 5:
                            row[econ_key] = ev

            # Age
            age_var = var_map.get("age")
            if age_var and age_var in raw_row.index:
                age_val = raw_row[age_var]
                if pd.notna(age_val):
                    age_f = float(age_val)
                    if var_map.get("age_is_birthyear"):
                        if 1900 <= age_f <= 2010:
                            row["age"] = year - age_f
                    elif 16 <= age_f <= 110:
                        row["age"] = age_f

            # Sex/Gender (1=Male, 2=Female across all years)
            sex_var = var_map.get("sex")
            if sex_var and sex_var in raw_row.index:
                sex_val = raw_row[sex_var]
                if pd.notna(sex_val):
                    sex_int = int(sex_val)
                    if sex_int == 1:
                        row["female"] = 0
                    elif sex_int == 2:
                        row["female"] = 1

            # Education (harmonized to 4 categories)
            edu_var = var_map.get("education")
            if edu_var and edu_var in raw_row.index:
                edu_val = raw_row[edu_var]
                if pd.notna(edu_val):
                    edu_int = int(edu_val)
                    recode = EDUCATION_RECODE.get(year, {})
                    if edu_int in recode:
                        row["education"] = recode[edu_int]

            # Income bracket
            inc_var = var_map.get("income")
            if inc_var and inc_var in raw_row.index:
                inc_val = raw_row[inc_var]
                if pd.notna(inc_val):
                    inc_int = int(inc_val)
                    if inc_int not in (9, 99) and inc_int >= 0:
                        row["income_bracket"] = inc_int

            # Ethnicity
            eth_euro_var = var_map.get("ethnicity_euro")
            eth_maori_var = var_map.get("ethnicity_maori")
            eth_id_var = var_map.get("ethnic_id")
            if eth_euro_var and eth_euro_var in raw_row.index:
                val = raw_row[eth_euro_var]
                if pd.notna(val):
                    row["ethnicity_euro"] = int(val) == 1
            if eth_maori_var and eth_maori_var in raw_row.index:
                val = raw_row[eth_maori_var]
                if pd.notna(val):
                    row["ethnicity_maori"] = int(val) == 1
            if eth_id_var and eth_id_var in raw_row.index:
                val = raw_row[eth_id_var]
                if pd.notna(val):
                    val_int = int(val)
                    row["ethnicity_euro"] = val_int == 1
                    row["ethnicity_maori"] = val_int == 2

            # Party like/dislike thermometers (0-10 scale)
            for like_key in ["party_like_lab", "party_like_nat", "party_like_grn",
                             "party_like_nzf", "party_like_act"]:
                like_var = var_map.get(like_key)
                if like_var and like_var in raw_row.index:
                    like_val = raw_row[like_var]
                    if pd.notna(like_val):
                        lf = float(like_val)
                        if 0 <= lf <= 10:
                            row[like_key] = lf

            respondents.append(row)

        year_df = pd.DataFrame(respondents)
        print(f"    Harmonized rows: {len(year_df)}")
        if "party_vote" in year_df.columns:
            valid_pv = year_df["party_vote"].notna().sum()
            print(f"    Valid party votes: {valid_pv}")
        all_respondents.append(year_df)

    combined = pd.concat(all_respondents, ignore_index=True)

    # Derived columns
    combined["vote_national"] = np.where(
        combined["party_vote"] == "National", 1,
        np.where(combined["party_vote"] == "Labour", 0, np.nan)
    )
    combined["age_group"] = pd.cut(
        combined["age"], bins=[0, 30, 45, 60, 100],
        labels=["18-29", "30-44", "45-59", "60+"], right=False,
    )
    edu_labels = {0: "No qualification", 1: "School", 2: "Post-school", 3: "University"}
    combined["education_label"] = combined["education"].map(edu_labels)

    combined.to_csv(OUTPUT_NZES, index=False)
    print(f"\nSaved harmonized NZES: {OUTPUT_NZES}")
    print(f"  Total respondents: {len(combined)}")
    print(f"  Columns: {list(combined.columns)}")
    print(f"  Election years: {sorted(combined['election_year'].unique())}")
    print(f"  Party vote distribution:")
    if "party_vote" in combined.columns:
        print(combined["party_vote"].value_counts().head(10).to_string())

    conn.close()
    return combined


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 1: Data Infrastructure")
    print("=" * 70)

    econ = extract_quarterly_economics()
    polls_econ = merge_polls_with_economics(econ)
    nzes = harmonize_nzes()

    print("\n" + "=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)
    print(f"  {OUTPUT_ECON}: {econ.shape}")
    print(f"  {OUTPUT_POLLS}: {polls_econ.shape}")
    print(f"  {OUTPUT_NZES}: {nzes.shape}")


if __name__ == "__main__":
    main()
