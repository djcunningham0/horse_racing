# Data Ingestion

Pipeline to convert raw Equibase data files into Parquet tables.

```bash
python -m data.ingest [--raw-dir DIR] [--output-dir DIR]
```

Defaults: `--raw-dir data/raw/equibase`, `--output-dir data/processed`.

## Sources and outputs

| Source | Parser | Output file | Grain |
|---|---|---|---|
| `2023_result_charts/*tch.xml` | `parse_results.parse_result_chart` | `results.parquet` | One row per horse per race |
| `2023_pps/*.zip` (each ZIP has one XML) | `parse_entries.parse_pps_zip` | `entries.parquet` | One row per horse per race |
| (same PPS ZIPs) | (same) | `past_performances.parquet` | One row per past performance per horse per race |
| (same PPS ZIPs) | (same) | `workouts.parquet` | One row per workout per horse per race |
| `Equibase Parameters.xlsx` | `parse_params.parse_track_codes` | `track_codes.parquet` | One row per track |

## Key fields

- **`race_id`** — synthetic key `{date}_{track}_R{number}` (e.g. `2023-04-29_CD_R5`), shared across results, entries, and past performances for joining.
- **Result charts** — race-level fields (purse, distance, surface, fractions, pace calls) plus per-horse fields (finish position, odds, speed rating, jockey/trainer, payoffs, point-of-call positions/lengths).
- **Entries (PPS)** — pre-race card data: morning line odds, weight carried, equipment, medication, class rating, age/sex restrictions, condition text.
- **Past performances** — each horse's recent race history: finish, speed figure, pace figures, odds, track condition, jockey/trainer, comments. Indexed `pp_index=1` for most recent.
- **Workouts** — date, track, distance, time, type, course, condition, ranking.

## Pipeline details

- All XML parsing uses `xml.etree.ElementTree`. Type conversions use `schema.safe_int` / `schema.safe_float` with `None` for missing/invalid values.
- Explicit Polars schemas are defined in `ingest.py` to avoid inference issues with sparse columns. Dynamic columns (e.g. `point_of_call_*` from result charts) are auto-detected across all rows.
- Past performances are sorted by date descending so `pp_index=1` is the most recent start.
- Odds parsing (`schema.parse_odds`) handles fractional (`5/1`), integer-coded PPS format (`5875` → `58.75`), and plain decimal strings.
- Parse errors are logged and skipped (not fatal) — error counts are reported at the end.
