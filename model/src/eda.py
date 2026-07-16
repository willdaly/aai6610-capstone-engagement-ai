"""Exploratory data analysis for the KDD Cup 1998 learning set.

Implements docs/eda_plan.md. Run it as a script:

    python model/src/eda.py

It writes docs/eda/column_inventory.csv and docs/figures/*.png, and prints the numbers
that docs/eda_findings.md quotes. The script is deterministic: the only randomness is
the train/test split, which is seeded through load_data.make_split.

Peeking rule (docs/eda_plan.md). Structural facts (shape, dtypes, missingness,
cardinality) use the full dataset. Anything touching TARGET_B or TARGET_D uses the
TRAINING split only. Every function below says which frame it takes, and every printed
number and figure title carries its frame. The test set stays untouched until final
model evaluation.

All data access goes through load_data. This module never reads the CSV itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend: this is a script, not a notebook, and Agg makes the PNG
# output identical whether or not a display is attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from load_data import (  # noqa: E402
    REPO_ROOT,
    TARGET_AMOUNT,
    TARGET_BINARY,
    load_raw,
    make_split,
)

FIGURES_DIR = REPO_ROOT / "docs" / "figures"
EDA_DIR = REPO_ROOT / "docs" / "eda"
INVENTORY_PATH = EDA_DIR / "column_inventory.csv"

# ---------------------------------------------------------------------------
# Figure styling
# ---------------------------------------------------------------------------
# Course visualization standards: trustworthy, effortless, elegant. Concretely that
# means one colorblind-safe palette across every figure, one accent color reserved for
# the thing the reader should look at, no chart junk, and axes that start at zero
# unless the figure states otherwise.
#
# Colors are from the Okabe-Ito colorblind-safe palette. BASE carries the data, ACCENT
# marks the one series or bar being argued about, and REFERENCE draws comparison lines.
BASE = "#0072B2"  # blue
ACCENT = "#D55E00"  # vermillion
SECONDARY = "#009E73"  # bluish green
REFERENCE = "#595959"  # neutral gray, for reference lines and annotations

FIGURE_DPI = 200

# Every figure the plan calls for. Kept as a module constant so the tests can assert
# the directory contents after a run without restating the list.
EXPECTED_FIGURES = (
    "target_d_distribution_responders.png",
    "missingness_top20.png",
    "response_rate_by_missingness_age_income.png",
    "age_distribution.png",
    "gender_income_counts.png",
    "giving_history_distributions.png",
    "census_exemplars.png",
    "response_rate_by_rfa2f.png",
    "response_rate_by_rfa2a.png",
    "response_rate_by_age_band.png",
    "response_rate_by_income.png",
    "correlation_giving_history.png",
)


def apply_style() -> None:
    """Set the rcParams every figure in this module inherits.

    Called once from main. Set here rather than per-figure so the figures cannot drift
    apart in styling as tasks get added.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,  # gridlines behind the data, never over it
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": REFERENCE,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, frame: str) -> None:
    """Apply the standard title/label treatment to one axes.

    Args:
        ax: The axes to style.
        title: A direct statement of the takeaway, not a restatement of the columns.
        xlabel: Axis label including units.
        ylabel: Axis label including units.
        frame: Which frame the numbers came from, e.g. "full dataset (n=95,412)". The
            peeking rule requires this on the figure itself, not just in the findings
            doc, so a figure pasted into the report carries its own provenance.
    """
    # Frame goes just under the title, small and gray: present for provenance, not
    # competing with the data for attention. Positioned in offset points rather than
    # axes fractions so it sits the same distance above the axes whatever the figure
    # height, and the title is padded past it by line count so a two-line title does
    # not land on top of it.
    frame_offset_points = 4
    ax.annotate(
        f"Frame: {frame}",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, frame_offset_points),
        textcoords="offset points",
        va="bottom",
        fontsize=8,
        color=REFERENCE,
    )
    title_line_height_points = 14
    ax.set_title(
        title,
        pad=frame_offset_points
        + title_line_height_points * (1 + title.count("\n")),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Write a figure to docs/figures/ at the standard dpi and close it.

    Closing matters: the script builds a dozen figures in one process and matplotlib
    warns once more than 20 are open at a time. Warnings are errors under pytest.ini.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Column group map
# ---------------------------------------------------------------------------
# docs/eda_plan.md carries a provisional group map and asks task 1 to verify it against
# the data. The map below is the corrected version; the corrections are recorded in the
# plan file and in docs/eda_findings.md. Assignment is by explicit membership rather
# than by name pattern wherever a pattern would be a guess.

GROUP_TARGET = "target"
GROUP_ID_ADMIN = "id_admin"
GROUP_GEOGRAPHY = "geography"
GROUP_DEMOGRAPHICS = "demographics"
GROUP_INTERESTS = "interests_overlay"
GROUP_CENSUS = "census"
GROUP_GIVING = "giving_history"
GROUP_PROMOTION = "promotion_history"

# The census block is one contiguous run of neighborhood statistics, POP901 through AC2
# inclusive. The plan guessed "roughly POP901 through the AC*/EC*/HC* ranges"; the
# EC*/HC* columns are inside the run, not after it, so the run is defined by endpoints.
CENSUS_BLOCK_START = "POP901"
CENSUS_BLOCK_END = "AC2"

# These three sit inside the POP901..AC2 run by position but are not neighborhood
# statistics: they are geographic area codes (metro area, ADI, DMA). Grouped as
# geography so nobody treats them as continuous percentages.
CENSUS_BLOCK_EXCLUSIONS = frozenset({"MSA", "ADI", "DMA"})

# Census percentages that sit outside the contiguous run, before it. The plan's map did
# not account for these; they are neighborhood percentages like the rest of the group.
CENSUS_OUTSIDE_BLOCK = (
    "MALEMILI",
    "MALEVET",
    "VIETVETS",
    "WWIIVETS",
    "LOCALGOV",
    "STATEGOV",
    "FEDGOV",
)

TARGET_COLUMNS = ("TARGET_B", "TARGET_D")

ID_ADMIN_COLUMNS = (
    "CONTROLN",  # row ID. Appears in the inventory and nowhere else (plan ground rule).
    "ODATEDW",  # date the donor was added to the file
    "OSOURCE",  # origin source code, 896 levels
    "TCODE",  # donor title code
    "MAILCODE",
    "PVASTATE",
    "NOEXCH",
    "RECINHSE",
    "RECP3",
    "RECPGVG",
    "RECSWEEP",
    "MDMAUD",
    "MDMAUD_R",
    "MDMAUD_F",
    "MDMAUD_A",
    "MAJOR",
    "DATASRCE",
    "LIFESRC",
    "PEPSTRFL",
    "SOLP3",
    "SOLIH",
    "HPHONE_D",
)

GEOGRAPHY_COLUMNS = (
    "STATE",
    "ZIP",
    "DOMAIN",
    "CLUSTER",
    "CLUSTER2",
    "GEOCODE",
    "GEOCODE2",
    "MSA",
    "ADI",
    "DMA",
)

DEMOGRAPHIC_COLUMNS = (
    "AGE",
    "AGEFLAG",
    "DOB",
    "GENDER",
    "INCOME",
    "HOMEOWNR",
    "WEALTH1",
    "WEALTH2",
    "CHILD03",
    "CHILD07",
    "CHILD12",
    "CHILD18",
    "NUMCHLD",
)

# Purchased overlay data: mail-order buying counts, publication subscriptions, and
# lifestyle interest flags. Not collected by the nonprofit, appended from a vendor.
INTERESTS_COLUMNS = (
    "HIT",
    "MBCRAFT",
    "MBGARDEN",
    "MBBOOKS",
    "MBCOLECT",
    "MAGFAML",
    "MAGFEM",
    "MAGMALE",
    "PUBGARDN",
    "PUBCULIN",
    "PUBHLTH",
    "PUBDOITY",
    "PUBNEWFN",
    "PUBPHOTO",
    "PUBOPP",
    "COLLECT1",
    "VETERANS",
    "BIBLE",
    "CATLG",
    "HOMEE",
    "PETS",
    "CDPLAY",
    "STEREO",
    "PCOWNERS",
    "PHOTO",
    "CRAFTS",
    "FISHER",
    "GARDENIN",
    "BOATS",
    "WALKER",
    "KIDSTUFF",
    "CARDS",
    "PLATES",
)

# Per-gift history plus the summary statistics derived from it.
GIVING_SUMMARY_COLUMNS = (
    "RAMNTALL",
    "NGIFTALL",
    "CARDGIFT",
    "MINRAMNT",
    "MINRDATE",
    "MAXRAMNT",
    "MAXRDATE",
    "LASTGIFT",
    "LASTDATE",
    "FISTDATE",
    "NEXTDATE",
    "TIMELAG",
    "AVGGIFT",
)

# Mailings sent, and the RFA (recency/frequency/amount) code as of each one.
PROMOTION_SUMMARY_COLUMNS = (
    "CARDPROM",
    "MAXADATE",
    "NUMPROM",
    "CARDPM12",
    "NUMPRM12",
    "RFA_2R",
    "RFA_2F",
    "RFA_2A",
)

# Prefixes for the repeating per-promotion column families.
GIVING_PREFIXES = ("RDATE_", "RAMNT_")
PROMOTION_PREFIXES = ("ADATE_", "RFA_")


def assign_group(column: str, census_block: frozenset[str]) -> str:
    """Return the group name for one column.

    Args:
        column: Column name.
        census_block: The POP901..AC2 run, already resolved against the real column
            order and already stripped of CENSUS_BLOCK_EXCLUSIONS.

    Returns:
        One of the GROUP_* constants.

    Raises:
        ValueError: If the column matches no group. Every one of the 481 columns must
            land in exactly one group, so an unassigned column is a bug in this map,
            not a column to quietly bucket as "other".
    """
    if column in TARGET_COLUMNS:
        return GROUP_TARGET
    # Explicit membership beats the prefix rules below: RFA_2R/2F/2A start with "RFA_"
    # but are listed in PROMOTION_SUMMARY_COLUMNS, and both resolve to the same group.
    if column in ID_ADMIN_COLUMNS:
        return GROUP_ID_ADMIN
    if column in GEOGRAPHY_COLUMNS:
        return GROUP_GEOGRAPHY
    if column in DEMOGRAPHIC_COLUMNS:
        return GROUP_DEMOGRAPHICS
    if column in INTERESTS_COLUMNS:
        return GROUP_INTERESTS
    if column in census_block or column in CENSUS_OUTSIDE_BLOCK:
        return GROUP_CENSUS
    if column in GIVING_SUMMARY_COLUMNS or column.startswith(GIVING_PREFIXES):
        return GROUP_GIVING
    if column in PROMOTION_SUMMARY_COLUMNS or column.startswith(PROMOTION_PREFIXES):
        return GROUP_PROMOTION
    raise ValueError(
        f"Column {column!r} matches no group in the map. Add it to a group in eda.py "
        "and record the correction in docs/eda_plan.md."
    )


def resolve_census_block(columns: list[str]) -> frozenset[str]:
    """Resolve the contiguous census run against the actual column order.

    Reads the endpoints out of the real frame rather than hardcoding positions, so the
    map breaks loudly if the file ever changes shape instead of silently mislabeling.

    Raises:
        ValueError: If an endpoint is absent or the run is inverted.
    """
    for endpoint in (CENSUS_BLOCK_START, CENSUS_BLOCK_END):
        if endpoint not in columns:
            raise ValueError(
                f"Census block endpoint {endpoint!r} is not a column in this frame."
            )

    start = columns.index(CENSUS_BLOCK_START)
    end = columns.index(CENSUS_BLOCK_END)
    if start >= end:
        raise ValueError(
            f"Census block endpoints are out of order: {CENSUS_BLOCK_START} at {start}, "
            f"{CENSUS_BLOCK_END} at {end}."
        )

    return frozenset(columns[start : end + 1]) - CENSUS_BLOCK_EXCLUSIONS


def blank_share(series: pd.Series) -> float:
    """Fraction of rows that are a blank or whitespace-only string.

    NaN is not blankness: pandas already counts that as missing. This measures the
    other kind, the kind that survives read_csv looking like data. Returns 0.0 for
    non-string columns.
    """
    if not pd.api.types.is_string_dtype(series) or pd.api.types.is_numeric_dtype(
        series
    ):
        return 0.0
    return float((series.str.strip() == "").mean())


# ---------------------------------------------------------------------------
# Task 1: inventory (full dataset)
# ---------------------------------------------------------------------------


def task1_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 481-row column inventory. Frame: full dataset.

    Structural facts only (dtype, missingness, cardinality, group), so the peeking rule
    permits the full dataset here.

    The plan asks for name, dtype, non-null count, % missing, cardinality, and group.
    pct_blank is added on top: task 3 has to find disguised missingness in categorical
    codes, and 65 of the 74 string columns carry blanks, so a NaN-only missingness
    column would understate the problem badly enough to mislead.

    Args:
        df: The full frame from load_raw.

    Returns:
        One row per column, in file order, with the columns written to
        docs/eda/column_inventory.csv.
    """
    columns = list(df.columns)
    census_block = resolve_census_block(columns)

    inventory = pd.DataFrame(
        {
            "column": columns,
            "dtype": [str(df[c].dtype) for c in columns],
            "non_null": [int(df[c].notna().sum()) for c in columns],
            "pct_missing": [round(float(df[c].isna().mean()) * 100, 4) for c in columns],
            "pct_blank": [round(blank_share(df[c]) * 100, 4) for c in columns],
            "nunique": [int(df[c].nunique()) for c in columns],
            "group": [assign_group(c, census_block) for c in columns],
        }
    )

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False)
    return inventory


def report_task1(inventory: pd.DataFrame) -> None:
    """Print the task 1 numbers that docs/eda_findings.md quotes."""
    print("=" * 78)
    print("TASK 1  Inventory. Frame: full dataset (n=95,412 rows, 481 columns).")
    print("=" * 78)
    print(f"\nWrote {INVENTORY_PATH.relative_to(REPO_ROOT)} ({len(inventory)} rows)\n")

    print("Columns per group:")
    by_group = inventory["group"].value_counts()
    for group, count in by_group.items():
        print(f"  {group:<18} {count:>4}")
    print(f"  {'TOTAL':<18} {by_group.sum():>4}\n")

    print("Columns per dtype:")
    for dtype, count in inventory["dtype"].value_counts().items():
        print(f"  {dtype:<18} {count:>4}")
    print()

    print("Top 20 columns by % missing (NaN only):")
    top_missing = inventory.nlargest(20, "pct_missing")
    for row in top_missing.itertuples():
        print(f"  {row.column:<12} {row.pct_missing:>6.2f}%   group={row.group}")
    print()

    print("Top 10 string columns by % blank (disguised missingness):")
    top_blank = inventory[inventory["pct_blank"] > 0].nlargest(10, "pct_blank")
    for row in top_blank.itertuples():
        print(f"  {row.column:<12} {row.pct_blank:>6.2f}%   group={row.group}")
    n_blank = int((inventory["pct_blank"] > 0).sum())
    n_str = int((inventory["dtype"] == "str").sum())
    print(f"\n  {n_blank} of {n_str} string columns contain blanks.\n")


# ---------------------------------------------------------------------------
# Task 2: targets (training split)
# ---------------------------------------------------------------------------


def task2_targets(train: pd.DataFrame, df: pd.DataFrame) -> None:
    """Describe both targets and write the TARGET_D histogram. Frame: training split.

    Args:
        train: Training split from make_split(df) with defaults.
        df: The full frame, used only to confirm the stratified split preserved the
            positive rate. That is a check on the split, not an analysis of the target,
            so reading the full rate here does not leak anything about the test set.
    """
    print("=" * 78)
    print(f"TASK 2  Targets. Frame: training split (n={len(train):,}).")
    print("=" * 78)

    positives = int(train[TARGET_BINARY].sum())
    train_rate = float(train[TARGET_BINARY].mean())
    full_rate = float(df[TARGET_BINARY].mean())

    print(f"\nTARGET_B, training split: {positives:,} positive of {len(train):,}")
    print(f"  training rate:     {train_rate * 100:.4f}%")
    print(f"  full dataset rate: {full_rate * 100:.4f}%  (stratification check)")
    print(f"  difference:        {abs(train_rate - full_rate) * 100:.4f} pp")
    print(f"  negatives per positive: {(1 - train_rate) / train_rate:.1f}")

    responders = train.loc[train[TARGET_BINARY] == 1, TARGET_AMOUNT]
    print(f"\nTARGET_D among responders only (n={len(responders):,}):")
    summary = responders.describe()
    for label in ("min", "25%", "50%", "75%", "max"):
        print(f"  {label:<5} ${summary[label]:>7.2f}")
    print(f"  mean  ${summary['mean']:>7.2f}")
    print(f"  std   ${summary['std']:>7.2f}")

    # Non-responders are documented as 0.0 rather than NaN. Confirm rather than assume:
    # a NaN here would silently drop rows from any later amount model.
    non_responders = train.loc[train[TARGET_BINARY] == 0, TARGET_AMOUNT]
    print(f"\n  Non-responders with a non-zero TARGET_D: {int((non_responders != 0).sum())}")
    print(f"  Non-responders with a NaN TARGET_D:      {int(non_responders.isna().sum())}")

    print("\n  Ten most common amounts (donations cluster on round numbers):")
    counts = responders.value_counts().head(10)
    for amount, count in counts.items():
        print(
            f"    ${amount:>6.2f}  {count:>4}  ({count / len(responders) * 100:>4.1f}% of responders)"
        )

    round_amounts = [5.0, 10.0, 15.0, 20.0, 25.0]
    n_round = int(responders.isin(round_amounts).sum())
    print(
        f"\n  Gave exactly $5/$10/$15/$20/$25: {n_round:,} of {len(responders):,} "
        f"({n_round / len(responders) * 100:.1f}%)"
    )
    n_multiple_of_5 = int((responders % 5 == 0).sum())
    print(
        f"  Gave an exact multiple of $5:    {n_multiple_of_5:,} of {len(responders):,} "
        f"({n_multiple_of_5 / len(responders) * 100:.1f}%)"
    )

    _figure_target_d(responders)
    print()


def _figure_target_d(responders: pd.Series) -> None:
    """Histogram of TARGET_D for responders. Frame: training split."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # $1 bins from 0 to the observed max. Fine enough that the spikes on round amounts
    # stay visible as spikes; coarser bins would smear them into the shape and hide the
    # very thing this figure is meant to show.
    top = float(responders.max())
    ax.hist(responders, bins=range(0, int(top) + 2), color=BASE)

    median = float(responders.median())
    ax.axvline(median, color=ACCENT, linewidth=2)
    ax.text(
        median + 2,
        ax.get_ylim()[1] * 0.92,
        f"median ${median:.0f}",
        color=ACCENT,
        fontsize=9,
        fontweight="bold",
    )

    style_axes(
        ax,
        title="Responder donations cluster on round amounts, and $10 is the mode",
        xlabel="TARGET_D, donation amount (US dollars)",
        ylabel="Responders",
        frame=f"training split, responders only (n={len(responders):,})",
    )
    # Axis runs to the true maximum ($200) rather than cropping the tail. The tail is
    # thin enough to be invisible at this scale, which is itself the honest picture.
    ax.set_xlim(0, top)
    save_figure(fig, "target_d_distribution_responders.png")


# ---------------------------------------------------------------------------
# Task 3: missing data
# ---------------------------------------------------------------------------


def task3_missing(df: pd.DataFrame, train: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Missingness: extent on the full dataset, target relationships on training.

    The two frames are kept apart deliberately. How much is missing is a structural
    fact and uses all 95,412 rows. Whether missingness predicts response is a statement
    about the target and uses the training split only.
    """
    print("=" * 78)
    print("TASK 3  Missing data. Frames: full dataset for extent, training for target.")
    print("=" * 78)

    top20 = inventory.nlargest(20, "pct_missing")
    _figure_missingness_top20(top20)

    print(f"\nColumns with any NaN:   {int((inventory['pct_missing'] > 0).sum())} of 481")
    print(f"Columns with any blank: {int((inventory['pct_blank'] > 0).sum())} of 481")
    print(
        f"Columns with neither:   "
        f"{int(((inventory['pct_missing'] == 0) & (inventory['pct_blank'] == 0)).sum())} of 481"
    )

    print("\nAGE and INCOME, observed values. Frame: full dataset.")
    for column in ("AGE", "INCOME"):
        series = df[column]
        print(
            f"  {column:<7} {series.isna().sum():>6,} missing "
            f"({series.isna().mean() * 100:5.2f}%)  "
            f"observed range {series.min():g} to {series.max():g}, "
            f"{series.nunique()} distinct"
        )
    print(f"  AGE values equal to 0: {int((df['AGE'] == 0).sum())} (no zero-coded ages)")

    print("\nResponse rate by missingness. Frame: training split.")
    overall = float(train[TARGET_BINARY].mean())
    print(f"  Overall training response rate: {overall * 100:.3f}%")

    rates = {}
    for column in ("AGE", "INCOME"):
        is_missing = train[column].isna()
        rate_missing = float(train.loc[is_missing, TARGET_BINARY].mean())
        rate_present = float(train.loc[~is_missing, TARGET_BINARY].mean())
        rates[column] = (rate_missing, rate_present, int(is_missing.sum()))

        # The plan makes the indicator-flag recommendation conditional on whether
        # missingness actually predicts response, so eyeballing a gap of a fifth of a
        # percentage point is not enough. Test it.
        contingency = pd.crosstab(is_missing, train[TARGET_BINARY])
        chi2, p_value, _dof, _expected = chi2_contingency(contingency)

        print(f"\n  {column}:")
        print(
            f"    missing (n={is_missing.sum():>6,}): {rate_missing * 100:.3f}% respond"
        )
        print(
            f"    present (n={(~is_missing).sum():>6,}): {rate_present * 100:.3f}% respond"
        )
        print(
            f"    difference: {(rate_missing - rate_present) * 100:+.3f} pp "
            f"(ratio {rate_missing / rate_present:.2f}x)"
        )
        print(
            f"    chi-square test of independence: chi2={chi2:.3f}, p={p_value:.4f} "
            f"-> {'predicts' if p_value < 0.05 else 'does not predict'} response at a=0.05"
        )

    _figure_missingness_response(rates, overall)

    # AGE is derived from DOB, and the derivation is where the missingness comes from.
    # Worth stating precisely because it means AGE and DOB carry one fact, not two.
    dob_zero_age_missing = int(((df["DOB"] == 0) & (df["AGE"].isna())).sum())
    age_missing_dob_present = int(((df["AGE"].isna()) & (df["DOB"] > 0)).sum())
    print(
        f"\n  AGE missingness tracks DOB=0 almost exactly. Frame: full dataset.\n"
        f"    DOB=0 and AGE missing:      {dob_zero_age_missing:,}\n"
        f"    AGE missing but DOB present: {age_missing_dob_present}\n"
        f"    DOB=0 but AGE present:       {int(((df['DOB'] == 0) & (df['AGE'].notna())).sum())}"
    )

    print("\nDisguised missingness in categorical codes. Frame: full dataset.")
    blanks = inventory[inventory["pct_blank"] > 0].nlargest(12, "pct_blank")
    print("  Worst 12 string columns by blank share:")
    for row in blanks.itertuples():
        print(f"    {row.column:<10} {row.pct_blank:>6.2f}%  group={row.group}")

    # Numeric columns can disguise missingness too, by coding it as a sentinel value.
    # The date fields are the ones at risk: a YYMM date of 0 is not a date.
    print("\n  Numeric columns coding missingness as 0 rather than NaN:")
    for column in ("DOB", "FISTDATE"):
        zeros = int((df[column] == 0).sum())
        print(
            f"    {column:<10} {zeros:>6,} rows are 0 ({zeros / len(df) * 100:.2f}%), "
            f"in a YYMM field otherwise ranging "
            f"{int(df.loc[df[column] > 0, column].min())} to {int(df[column].max())}"
        )
    print()


def _figure_missingness_top20(top20: pd.DataFrame) -> None:
    """Horizontal bar of the 20 most-missing columns. Frame: full dataset."""
    fig, ax = plt.subplots(figsize=(9, 7))

    ordered = top20.sort_values("pct_missing")
    positions = range(len(ordered))
    ax.barh(list(positions), ordered["pct_missing"], color=BASE)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(ordered["column"])
    for y, value in zip(positions, ordered["pct_missing"]):
        ax.text(value + 0.6, y, f"{value:.2f}%", va="center", fontsize=8, color=REFERENCE)

    style_axes(
        ax,
        title="The most-missing columns are all per-mailing response records,\nwhere missing means 'did not give', not 'unknown'",
        xlabel="Rows missing (%)",
        ylabel="",
        frame="full dataset (n=95,412)",
    )
    ax.set_xlim(0, 108)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "missingness_top20.png")


def _figure_missingness_response(
    rates: dict[str, tuple[float, float, int]], overall: float
) -> None:
    """Response rate for missing vs present AGE/INCOME. Frame: training split."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    values = []
    colors = []
    for column, (rate_missing, rate_present, _n) in rates.items():
        labels.extend([f"{column}\nmissing", f"{column}\npresent"])
        values.extend([rate_missing * 100, rate_present * 100])
        # Accent the missing bars: they are the comparison the figure exists to make.
        colors.extend([ACCENT, BASE])

    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"{value:.2f}%",
            ha="center",
            fontsize=9,
        )

    ax.axhline(overall * 100, color=REFERENCE, linestyle="--", linewidth=1.2)
    # Below the line, not above: above collides with the bar value labels, which all
    # sit within a quarter of a point of the reference.
    ax.text(
        len(labels) - 0.45,
        overall * 100 - 0.42,
        f"overall {overall * 100:.2f}%",
        ha="right",
        fontsize=8,
        color=REFERENCE,
    )

    style_axes(
        ax,
        title="Missingness in AGE or INCOME does not predict response:\nthe two gaps are small, point in opposite directions, and are not significant",
        xlabel="",
        ylabel="Response rate (% giving to the campaign)",
        frame="training split (n=76,329)",
    )
    # Y axis starts at zero and runs well past the bars. Zooming in on the 4.8-5.3%
    # range would turn a null result into an apparent effect, which is the specific
    # dishonesty this figure has to avoid: the bars genuinely are near-identical.
    ax.set_ylim(0, max(values) * 1.6)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "response_rate_by_missingness_age_income.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run every task in plan order, writing tables and figures under docs/."""
    apply_style()

    df = load_raw()
    train, _test = make_split(df)
    print(
        f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns. "
        f"Training split: {train.shape[0]:,} rows "
        f"({train[TARGET_BINARY].mean() * 100:.2f}% positive).\n"
    )

    inventory = task1_inventory(df)
    report_task1(inventory)
    task2_targets(train, df)
    task3_missing(df, train, inventory)


if __name__ == "__main__":
    main()
