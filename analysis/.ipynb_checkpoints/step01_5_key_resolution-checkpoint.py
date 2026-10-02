# # STEP 1.5 - KEY RESOLUTION
# A. Resolve duplicate train_no
# B. Identify the 1 unmatched schedule station
# C. Identify the 63 unmatched delay stations
# D. Globally verify (date, train_no, station_no)
# E. Determine the exact grain of FACT_DELAY
#
# Engine: DuckDB (38M rows is handled out-of-core, no chunk-boundary problem).
import json
import duckdb
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 120)

BASE_DIR   = Path.cwd().parent
DATA_DIR   = BASE_DIR / "data_sets"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
TMP_DIR    = RESULT_DIR / "duckdb_tmp"
TMP_DIR.mkdir(exist_ok=True)

con = duckdb.connect(str(RESULT_DIR / "step01_5.duckdb"))
con.execute("SET memory_limit='6GB'")            # lower this if your laptop has < 8 GB RAM
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
con.execute("SET preserve_insertion_order=false")

FINDINGS = {}          # everything that matters is collected here -> saved as JSON at the end


def q(sql):
    return con.execute(sql).df()


def save(df, name):
    df.to_csv(RESULT_DIR / name, index=False)
    print(f"  -> saved results/{name}")


# %% [markdown]
# ## 0. Load tables into DuckDB (delay is loaded once and cached in the .duckdb file)

# %%
def load(name, fname, types=None):
    path = (DATA_DIR / fname).as_posix()
    t = f", types={types}" if types else ""
    con.execute(
        f"CREATE OR REPLACE TABLE {name} AS "
        f"SELECT * FROM read_csv('{path}', header=true{t})"
    )


load("train_details",      "train_details.csv",      {"train_no": "BIGINT"})
load("station_full_names", "station_full_names.csv")
load("schedule",           "combined_schedule.csv",  {"train_no": "BIGINT", "station_no": "INTEGER"})

already = con.execute(
    "SELECT count(*) FROM information_schema.tables WHERE table_name='delay'"
).fetchone()[0]
if not already:
    print("Loading combined_delay.csv (one-time, a few minutes)...")
    load("delay", "combined_delay.csv",
         {"date": "DATE", "station_no": "INTEGER", "delay": "DOUBLE", "train_no": "BIGINT"})

for t in ["train_details", "station_full_names", "schedule", "delay"]:
    print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])

# Helper tables used repeatedly
con.execute("CREATE OR REPLACE TABLE sched_stations AS SELECT DISTINCT station_name FROM schedule")
con.execute("CREATE OR REPLACE TABLE delay_stations AS SELECT DISTINCT station_name FROM delay")

# %% [markdown]
# ## A. Resolve duplicate train_no
# Question: are the 272 repeated train_no values the *same train* listed under two type_codes,
# or genuinely different trains that share a number?

# %%
print("=== A1. Shape of the duplication ===")
shape = q("""
SELECT count(*) AS dup_train_nos,
       sum(n > 2)        AS with_more_than_2_rows,
       sum(names > 1)    AS with_different_names
FROM (SELECT train_no, count(*) n, count(DISTINCT train_name) names
      FROM train_details GROUP BY train_no HAVING count(*) > 1)
""")
print(shape.to_string(index=False))

print("\n=== A2. Which type_code combinations occur among duplicates? ===")
combos = q("""
SELECT combo, count(*) AS n_train_no
FROM (SELECT train_no, string_agg(DISTINCT type_code, ' + ' ORDER BY type_code) AS combo
      FROM train_details GROUP BY train_no HAVING count(*) > 1)
GROUP BY combo ORDER BY n_train_no DESC
""")
print(combos.to_string(index=False))
save(combos, "step15_A_duplicate_type_combos.csv")

print("\n=== A3. Are duplicated train numbers actually used in schedule / delay? ===")
usage = q("""
WITH d AS (SELECT train_no FROM train_details GROUP BY train_no HAVING count(*) > 1)
SELECT count(*) AS dup_train_nos,
       sum(CASE WHEN train_no IN (SELECT train_no FROM schedule) THEN 1 ELSE 0 END) AS in_schedule,
       sum(CASE WHEN train_no IN (SELECT train_no FROM delay)    THEN 1 ELSE 0 END) AS in_delay
FROM d
""")
print(usage.to_string(index=False))

# Resolution rule (hypothesis, TESTED by A2):
#   PRM-TRAINS behaves like an overlay tag ("premium/special") on top of a base service type.
#   -> keep the base type (EXP / SF / SHT / RAJ / ...) as primary_type_code,
#      keep the PRM info as a boolean flag.  If a group has >1 distinct non-PRM types -> 'REVIEW'.
con.execute("""
CREATE OR REPLACE TABLE dim_train_resolved AS
SELECT train_no,
       min(train_name)                                   AS train_name,
       count(DISTINCT train_name)                        AS n_names_raw,
       CASE
         WHEN count(DISTINCT type_code) FILTER (WHERE type_code <> 'PRM-TRAINS') = 1
              THEN min(type_code)        FILTER (WHERE type_code <> 'PRM-TRAINS')
         WHEN count(DISTINCT type_code) FILTER (WHERE type_code <> 'PRM-TRAINS') = 0
              THEN 'PRM-TRAINS'
         ELSE 'REVIEW'
       END                                               AS primary_type_code,
       bool_or(type_code = 'PRM-TRAINS')                 AS is_premium_special,
       string_agg(DISTINCT type_code, '|' ORDER BY type_code) AS raw_type_codes,
       count(*)                                          AS raw_row_count
FROM train_details
GROUP BY train_no
""")

res = q("""
SELECT count(*)                                         AS rows,
       count(DISTINCT train_no)                         AS unique_train_no,
       sum(CASE WHEN primary_type_code='REVIEW' THEN 1 ELSE 0 END) AS needs_manual_review,
       sum(CASE WHEN n_names_raw > 1 THEN 1 ELSE 0 END)            AS name_conflicts,
       sum(CASE WHEN is_premium_special THEN 1 ELSE 0 END)         AS premium_flagged
FROM dim_train_resolved
""")
print("\n=== A4. Resolved train dimension ===")
print(res.to_string(index=False))

print("\nprimary_type_code distribution after resolution:")
print(q("SELECT primary_type_code, count(*) AS n FROM dim_train_resolved GROUP BY 1 ORDER BY n DESC").to_string(index=False))

review = q("SELECT * FROM dim_train_resolved WHERE primary_type_code='REVIEW' OR n_names_raw>1")
if len(review):
    print("\n!! Rows needing manual decision:")
    print(review.to_string(index=False))
    save(review, "step15_A_train_manual_review.csv")

save(q("SELECT * FROM dim_train_resolved ORDER BY train_no"), "dim_train_resolved.csv")

FINDINGS["A"] = {
    "raw_rows": 8992,
    "dup_train_nos": int(shape.dup_train_nos[0]),
    "dup_with_different_names": int(shape.with_different_names[0]),
    "resolved_rows": int(res.rows[0]),
    "resolved_unique_train_no": int(res.unique_train_no[0]),
    "needs_manual_review": int(res.needs_manual_review[0]),
    "rule": "primary_type_code = the single non-PRM type; PRM kept as is_premium_special flag",
}
assert res.rows[0] == res.unique_train_no[0], "train_no still not unique after resolution!"

# %% [markdown]
# ## B + C. Unmatched stations (schedule: 1, delay: 63)
# Each unmatched code is classified:
# - FORMAT_VARIANT        : matches master after upper/trim  -> safe auto-fix
# - FULLNAME_USED_AS_CODE : the "code" is really a station_full_name in master -> mapping candidate
# - NEAR_MATCH_REVIEW     : Levenshtein <= 1 to a master code -> NEVER auto-merge
#                           (codes like ABC/ABD are usually *different* stations), manual check
# - MISSING_FROM_MASTER   : genuinely absent from master -> keep, add as 'unknown-metadata' member

# %%
def diagnose_unmatched(tag, src_table):
    con.execute(f"""
    CREATE OR REPLACE TABLE unmatched_{tag} AS
    SELECT s.station_name,
           count(*)                  AS rows_in_source,
           count(DISTINCT s.train_no) AS n_trains
    FROM {src_table} s
    LEFT JOIN station_full_names m ON s.station_name = m.station_name
    WHERE m.station_name IS NULL
    GROUP BY s.station_name
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE diag_{tag} AS
    WITH u AS (SELECT * FROM unmatched_{tag}),
    fmt AS (
        SELECT u.station_name, any_value(m.station_name) AS fmt_match
        FROM u JOIN station_full_names m
          ON upper(trim(u.station_name)) = upper(trim(m.station_name))
        GROUP BY 1),
    fn AS (
        SELECT u.station_name, any_value(m.station_name) AS fullname_match
        FROM u JOIN station_full_names m
          ON upper(trim(u.station_name)) = upper(trim(m.station_full_name))
        GROUP BY 1),
    fz AS (
        SELECT u.station_name,
               arg_max(m.station_name, jaro_winkler_similarity(u.station_name, m.station_name)) AS nearest_code,
               max(jaro_winkler_similarity(u.station_name, m.station_name))                     AS jw
        FROM u CROSS JOIN station_full_names m
        GROUP BY 1),
    base AS (
        SELECT u.*, fmt.fmt_match, fn.fullname_match, fz.nearest_code, round(fz.jw, 3) AS jw,
               levenshtein(u.station_name, fz.nearest_code) AS lev,
               u.station_name IN (SELECT station_name FROM sched_stations) AS in_schedule,
               u.station_name IN (SELECT station_name FROM delay_stations) AS in_delay
        FROM u
        LEFT JOIN fmt USING (station_name)
        LEFT JOIN fn  USING (station_name)
        LEFT JOIN fz  USING (station_name))
    SELECT *,
           CASE WHEN fmt_match      IS NOT NULL THEN 'FORMAT_VARIANT'
                WHEN fullname_match IS NOT NULL THEN 'FULLNAME_USED_AS_CODE'
                WHEN lev <= 1                   THEN 'NEAR_MATCH_REVIEW'
                ELSE 'MISSING_FROM_MASTER' END AS category
    FROM base
    ORDER BY rows_in_source DESC
    """)
    return q(f"SELECT * FROM diag_{tag}")


# ---------- B ----------
print("=== B. Unmatched station(s) in combined_schedule ===")
diag_sched = diagnose_unmatched("schedule", "schedule")
print(diag_sched.to_string(index=False))
save(diag_sched, "step15_B_unmatched_schedule_station.csv")

print("\nWhere does it sit in the routes? (first 15 rows)")
print(q("""
SELECT s.train_no, s.station_no, s.station_name, s.distance_from_origin,
       s.arrival_time, s.departure_time
FROM schedule s JOIN unmatched_schedule u USING (station_name)
ORDER BY s.train_no, s.station_no LIMIT 15
""").to_string(index=False))

print("\nTrain(s) affected and their type:")
print(q("""
SELECT t.primary_type_code, count(DISTINCT s.train_no) AS trains,
       min(s.train_no) AS example_train
FROM schedule s JOIN unmatched_schedule u USING (station_name)
JOIN dim_train_resolved t ON t.train_no = s.train_no
GROUP BY 1
""").to_string(index=False))

FINDINGS["B"] = {
    "n_unmatched_codes": int(len(diag_sched)),
    "codes": diag_sched.to_dict(orient="records"),
}

# ---------- C ----------
print("\n=== C. Unmatched stations in combined_delay ===")
diag_delay = diagnose_unmatched("delay", "delay")
print(f"unmatched delay station codes: {len(diag_delay)}")

cat = q("""
SELECT category, count(*) AS codes, sum(rows_in_source) AS delay_rows,
       sum(CASE WHEN in_schedule THEN 1 ELSE 0 END) AS also_in_schedule
FROM diag_delay GROUP BY category ORDER BY delay_rows DESC
""")
print("\nBy category:")
print(cat.to_string(index=False))

impact = q("""
SELECT d.category,
       count(*)        AS delay_rows,
       count(x.delay)  AS delay_observed,
       round(100.0 * count(*) / (SELECT count(*) FROM delay), 4) AS pct_of_all_delay_rows
FROM delay x JOIN diag_delay d USING (station_name)
GROUP BY d.category ORDER BY delay_rows DESC
""")
print("\nImpact on fact table:")
print(impact.to_string(index=False))

print("\nAll 63 codes (top 40 by rows):")
print(diag_delay.head(40).to_string(index=False))
save(diag_delay, "step15_C_unmatched_delay_stations.csv")

# Station resolution map (input for ETL)
resmap = pd.concat(
    [diag_sched.assign(source="schedule"), diag_delay.assign(source="delay")],
    ignore_index=True,
)
save(resmap, "station_resolution_map.csv")

FINDINGS["C"] = {
    "n_unmatched_codes": int(len(diag_delay)),
    "by_category": cat.to_dict(orient="records"),
    "impact": impact.to_dict(orient="records"),
}

# %% [markdown]
# ## D. Global verification of (date, train_no, station_no)
# Single DuckDB hash-aggregate over all 38.4M rows -> no chunk-boundary blind spot.

# %%
total_delay_rows = con.execute("SELECT count(*) FROM delay").fetchone()[0]

print("=== D1. Global uniqueness of (date, train_no, station_no) ===")
dk = q("""
SELECT count(*)                       AS duplicate_key_groups,
       coalesce(sum(c - 1), 0)::BIGINT AS surplus_rows
FROM (SELECT date, train_no, station_no, count(*) AS c
      FROM delay GROUP BY date, train_no, station_no HAVING count(*) > 1)
""")
print(f"total rows: {total_delay_rows:,}")
print(dk.to_string(index=False))

if dk.duplicate_key_groups[0] > 0:
    print("\n!! Duplicates exist. Are they identical or conflicting?")
    kind = q("""
    SELECT CASE WHEN n_vals = 1 THEN 'identical_delay' ELSE 'conflicting_delay' END AS kind,
           count(*) AS groups
    FROM (SELECT date, train_no, station_no,
                 count(DISTINCT coalesce(delay, -999999)) AS n_vals
          FROM delay GROUP BY date, train_no, station_no HAVING count(*) > 1)
    GROUP BY 1
    """)
    print(kind.to_string(index=False))
    sample = q("""
    SELECT d.* FROM delay d
    JOIN (SELECT date, train_no, station_no FROM delay
          GROUP BY date, train_no, station_no HAVING count(*) > 1 LIMIT 10) k
    USING (date, train_no, station_no)
    ORDER BY date, train_no, station_no
    """)
    print(sample.to_string(index=False))
    save(sample, "step15_D_duplicate_key_sample.csv")

print("\n=== D2. Alternative key (date, train_no, station_name) - detects loops/repeat visits ===")
dk2 = q("""
SELECT count(*) AS duplicate_key_groups
FROM (SELECT date, train_no, station_name FROM delay
      GROUP BY date, train_no, station_name HAVING count(*) > 1)
""")
print(dk2.to_string(index=False))

print("\n=== D3. Schedule side: is (train_no, station_no) a key of the route? ===")
sk = q("""
SELECT (SELECT count(*) FROM schedule) AS rows,
       (SELECT count(*) FROM (SELECT DISTINCT train_no, station_no FROM schedule)) AS distinct_train_stop,
       (SELECT count(*) FROM (SELECT train_no, station_name FROM schedule
                              GROUP BY ALL HAVING count(*) > 1)) AS trains_visiting_station_twice,
       (SELECT count(*) FROM (SELECT train_no FROM schedule GROUP BY train_no
                              HAVING max(station_no) <> count(*) OR min(station_no) <> 1)) AS non_contiguous_routes
""")
print(sk.to_string(index=False))

print("\n=== D4. Does delay agree with schedule on (train_no, station_no) -> station_name? ===")
con.execute("""
CREATE OR REPLACE TABLE delay_triples AS
SELECT DISTINCT train_no, station_no, station_name FROM delay
""")
cf = q("""
SELECT count(*) AS delay_triples,
       sum(CASE WHEN s.train_no IS NULL THEN 1 ELSE 0 END) AS no_schedule_match,
       sum(CASE WHEN s.train_no IS NOT NULL AND s.station_name <> d.station_name THEN 1 ELSE 0 END) AS station_name_mismatch,
       (SELECT count(*) FROM (SELECT DISTINCT train_no, station_no FROM delay_triples)) AS distinct_train_stop_in_delay
FROM delay_triples d
LEFT JOIN schedule s ON d.train_no = s.train_no AND d.station_no = s.station_no
""")
print(cf.to_string(index=False))

print("\nstation_no values present in delay but not in schedule (121 vs 119 distinct):")
extra_no = q("""
SELECT station_no, count(*) AS delay_rows, count(DISTINCT train_no) AS trains
FROM delay WHERE station_no IN (SELECT station_no FROM delay EXCEPT SELECT station_no FROM schedule)
GROUP BY 1 ORDER BY 1
""")
print(extra_no.to_string(index=False) if len(extra_no) else "  none")

print("\nOrphan delay triples (no schedule match) - sample:")
orph = q("""
SELECT d.train_no, d.station_no, d.station_name,
       d.train_no IN (SELECT train_no FROM schedule) AS train_in_schedule
FROM delay_triples d
LEFT JOIN schedule s ON d.train_no = s.train_no AND d.station_no = s.station_no
WHERE s.train_no IS NULL LIMIT 20
""")
print(orph.to_string(index=False) if len(orph) else "  none")
if len(orph):
    save(orph, "step15_D_orphan_delay_triples_sample.csv")

FINDINGS["D"] = {
    "total_rows": int(total_delay_rows),
    "dup_key_groups_date_train_stationno": int(dk.duplicate_key_groups[0]),
    "surplus_rows": int(dk.surplus_rows[0]),
    "dup_key_groups_date_train_stationname": int(dk2.duplicate_key_groups[0]),
    "schedule_checks": sk.to_dict(orient="records")[0],
    "delay_vs_schedule": cf.to_dict(orient="records")[0],
    "extra_station_no_in_delay": extra_no.to_dict(orient="records"),
}

# %% [markdown]
# ## E. Exact grain of FACT_DELAY
# Beyond key uniqueness we must know: is a (date, train_no) "trip" complete?
# Does `date` mean journey-start date (needed for multi-day trains)?
# What does a NULL delay mean (cancelled / not scraped / not reported)?

# %%
con.execute("""
CREATE OR REPLACE TABLE sched_trip AS
SELECT train_no, count(*) AS sched_stops, max(arrival_day) AS max_day
FROM schedule GROUP BY train_no
""")
con.execute("""
CREATE OR REPLACE TABLE trip AS
SELECT date, train_no, count(*) AS n_rows, count(delay) AS n_obs
FROM delay GROUP BY date, train_no
""")

print("=== E1. Trip completeness: rows per (date, train_no) vs scheduled stops ===")
cov = q("""
SELECT CASE WHEN s.train_no IS NULL        THEN '4_no_schedule'
            WHEN t.n_rows =  s.sched_stops THEN '1_full'
            WHEN t.n_rows <  s.sched_stops THEN '2_partial'
            ELSE                                 '3_more_than_schedule' END AS coverage,
       s.max_day > 1                        AS multi_day_train,
       count(*)                             AS trips,
       sum(t.n_rows)                        AS delay_rows
FROM trip t LEFT JOIN sched_trip s USING (train_no)
GROUP BY ALL ORDER BY 1, 2
""")
print(cov.to_string(index=False))
save(cov, "step15_E_trip_coverage.csv")

print("\n=== E2. What does a missing delay mean at trip level? ===")
miss = q("""
SELECT CASE WHEN n_obs = 0 THEN 'all_stops_missing'
            WHEN n_obs = n_rows THEN 'no_stop_missing'
            ELSE 'partially_missing' END AS trip_type,
       count(*) AS trips, sum(n_rows) AS rows
FROM trip GROUP BY 1 ORDER BY trips DESC
""")
print(miss.to_string(index=False))
save(miss, "step15_E_trip_missing_types.csv")

print("\nTrip counts per date (min / median / max):")
print(q("""
SELECT min(c) AS min_trips, median(c) AS median_trips, max(c) AS max_trips,
       count(*) AS dates
FROM (SELECT date, count(*) c FROM trip GROUP BY date)
""").to_string(index=False))

print("\nDays of data per train (distribution):")
print(q("""
SELECT min(d) AS min_days, quantile_cont(d, 0.25) AS q1, median(d) AS median_days,
       quantile_cont(d, 0.75) AS q3, max(d) AS max_days, count(*) AS trains
FROM (SELECT train_no, count(DISTINCT date) d FROM delay GROUP BY train_no)
""").to_string(index=False))

print("\n=== E3. Grain verdict ===")
dup_ok   = FINDINGS["D"]["surplus_rows"] == 0
conf_ok  = (FINDINGS["D"]["delay_vs_schedule"]["no_schedule_match"] == 0
            and FINDINGS["D"]["delay_vs_schedule"]["station_name_mismatch"] == 0)
loop_ok  = FINDINGS["D"]["dup_key_groups_date_train_stationname"] == 0

verdict = {
    "candidate_key": ["date", "train_no", "station_no"],
    "globally_unique": bool(dup_ok),
    "consistent_with_schedule": bool(conf_ok),
    "station_name_also_unique_within_trip": bool(loop_ok),
}
print(json.dumps(verdict, indent=2))

if dup_ok:
    print("""
GRAIN (confirmed if the three flags above are True):
  One row of FACT_DELAY = one train (train_no) on one service date (date)
  at one scheduled stop of its route (station_no = stop sequence).
  Physical station is an attribute of the stop (via schedule), NOT part of the key.
  delay is NULLABLE: NULL = 'not observed', never replaced by 0.
""")
else:
    print("Key is NOT unique -> add a tiebreaker/dedup rule before finalizing the grain (see D1 samples).")

FINDINGS["E"] = {
    "coverage": cov.to_dict(orient="records"),
    "missing_types": miss.to_dict(orient="records"),
    "verdict": verdict,
}

# %%
with open(RESULT_DIR / "step15_findings.json", "w") as f:
    json.dump(FINDINGS, f, indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Saved results/step15_findings.json")
print("Paste the printed output of sections A-E back here and I'll write the decisions + documentation entry.")
