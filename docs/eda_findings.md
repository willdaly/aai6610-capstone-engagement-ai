# EDA Findings — KDD Cup 1998 Learning Set

What the exploratory analysis specified in `docs/eda_plan.md` actually found. Organized
by the plan's task numbers. Every number here comes from a run of
`python model/src/eda.py`; nothing is estimated or carried over from documentation.

The dataset is a 1998 direct-mail campaign from a national veterans nonprofit, used as
a public proxy for constituent engagement. It is not Sturge-Weber Foundation data, and
this analysis is not affiliated with or endorsed by the Foundation.

**Frames.** Per the plan's peeking rule, structural facts (shape, dtypes, missingness,
cardinality) are computed on the full dataset of 95,412 rows. Anything involving
`TARGET_B` or `TARGET_D` is computed on the training split only (76,329 rows, from
`make_split(df)` with course defaults: 80/20 stratified, `random_state=2026`). The test
split of 19,083 rows is untouched and stays that way until final model evaluation.
Every claim below states its frame.

---

## 1. Inventory

**Frame: full dataset (95,412 rows, 481 columns).** Structural facts only.

The full inventory is `docs/eda/column_inventory.csv`, one row per column, 481 rows.
It carries the column name, pandas dtype, non-null count, % missing (NaN), % blank
(whitespace-only strings), cardinality, and the assigned group.

### Columns per group

| Group | Columns | What it holds |
| --- | ---: | --- |
| `census` | 290 | Neighborhood statistics from the 1990 census, appended by zip |
| `giving_history` | 57 | Per-gift dates and amounts, plus derived summaries |
| `promotion_history` | 54 | Mailings sent and the RFA code as of each |
| `interests_overlay` | 33 | Vendor-appended mail-order and lifestyle indicators |
| `id_admin` | 22 | Row ID, origin codes, donor flags |
| `demographics` | 13 | Age, gender, income, wealth, household composition |
| `geography` | 10 | State, zip, and area/cluster codes |
| `target` | 2 | `TARGET_B`, `TARGET_D` |
| **Total** | **481** | |

The single most important structural fact: 290 of 481 columns (60.3%) are census
neighborhood statistics, and only 13 (2.7%) describe the individual constituent. The
dataset is mostly about where people live, not who they are. Anything the model learns
from the census block is a statement about a neighborhood, which is exactly the kind of
inference that needs the fairness checks the proposal commits to.

### Columns per dtype

| dtype | Columns |
| --- | ---: |
| `int64` | 310 |
| `float64` | 97 |
| `str` | 74 |

No column loaded as `object`. `load_data.load_raw` declares `NOEXCH` as a string and
escalates any other mixed-type column to an error, so these dtypes are inference that
has been checked, not inference that was trusted.

### Verification of the group map

The plan's map was provisional and the data disagreed with it in five places. The
corrections are recorded in `docs/eda_plan.md`, marked **[corrected]**, and implemented
in `model/src/eda.py`, where the group assignment raises rather than bucketing an
unmatched column as "other". All 481 columns match exactly one group.

1. **The census run ends at `AC2`, and the `EC*`/`HC*` families are inside it, not
   after it.** The plan guessed "roughly `POP901` through the `AC*`/`EC*`/`HC*`
   ranges". `POP901` through `AC2` is one contiguous run of 286 columns ending
   immediately before `ADATE_2`. The group is defined by those endpoints.
2. **`MSA`, `ADI`, and `DMA` sit inside the run but are not neighborhood statistics.**
   They are geographic area codes (`MSA` ranges 0 to 9,360 across 298 levels). Grouping
   them with the percentages would invite treating an area code as a continuous
   quantity. They are excluded from the census block and assigned to geography.
3. **Seven census percentages sit outside the run.** `MALEMILI`, `MALEVET`,
   `VIETVETS`, `WWIIVETS`, `LOCALGOV`, `STATEGOV`, and `FEDGOV` are at positions 43-49,
   in the middle of the overlay columns, and are percentages on a 0-99 scale like the
   rest of the census group. The plan's map missed them. With those seven added and
   three area codes removed, the census group is 290 columns.
4. **Geography and purchased overlay are their own groups.** The plan folded zip and
   state into admin and had nowhere to put the 33 mail-order and lifestyle columns.
   Overlay data was appended by a vendor rather than collected by the nonprofit, and
   that provenance difference matters for both modeling and the ethics discussion.
5. **`RFA_2R` is constant.** The plan says to prefer the precomputed `RFA_2R`,
   `RFA_2F`, `RFA_2A` components. `RFA_2R` is `L` for all 95,412 rows and carries no
   information. Only `RFA_2F` and `RFA_2A` are usable, so task 5 charts those two and
   `RFA_2R` is reported under task 6 as a near-constant column.

### Top 20 columns by % missing (NaN)

| Column | % missing | Column | % missing |
| --- | ---: | --- | ---: |
| `RDATE_5` | 99.99 | `RDATE_23` | 91.76 |
| `RAMNT_5` | 99.99 | `RAMNT_23` | 91.76 |
| `RDATE_3` | 99.75 | `RDATE_20` | 91.73 |
| `RAMNT_3` | 99.75 | `RAMNT_20` | 91.73 |
| `RDATE_4` | 99.71 | `RDATE_7` | 90.68 |
| `RAMNT_4` | 99.71 | `RAMNT_7` | 90.68 |
| `RDATE_6` | 99.19 | `RDATE_17` | 90.15 |
| `RAMNT_6` | 99.19 | `RAMNT_17` | 90.15 |
| `RDATE_15` | 92.39 | `RDATE_21` | 90.03 |
| `RAMNT_15` | 92.39 | `RAMNT_21` | 90.03 |

Every column in the top 20 is an `RDATE_*`/`RAMNT_*` pair from the giving-history
group, and the pairs match exactly: `RDATE_5` and `RAMNT_5` are both 99.99% missing.
That is structure, not damage. These columns record the date and amount of the response
to one specific historical mailing, so a row is missing whenever that constituent did
not give to that mailing, which is nearly always. The missingness *is* the information,
and the honest reading is "did not respond", not "value unknown".

Two consequences. First, no imputation scheme belongs anywhere near these columns; task
6 records the handling recommendation. Second, this top-20 list is a poor summary of
where the dataset actually has gaps: `AGE` (24.80%) and `INCOME` (22.31%) do not appear
in it, and neither does any column whose missingness is genuine ignorance rather than a
non-event. Task 3 takes that up.

### Disguised missingness

65 of the 74 string columns contain blank or whitespace-only values, and none of them
appear in the NaN-based top 20 above, because `read_csv` reads a space as data. The
worst are `RECPGVG` (99.88% blank), `SOLP3` (99.81%), `MAJOR` (99.69%), `PLATES`
(99.41%), and `HOMEE` (99.07%). The plan cited `NOEXCH` as the example of the pattern;
`NOEXCH` is in fact the mildest case in the dataset, with 7 blank rows out of 95,412.
The inventory therefore reports `pct_blank` alongside `pct_missing`, and any statement
about missingness in the report has to account for both. Task 3 covers the substance.
