"""Copy the archive tables camanchaca reads from the heat-wave-tracker's Neon project into
camanchaca's own, so the two apps never share a database.

Idempotent: tables are created if missing (columns, primary key and unique constraints taken
from the source) and rows already present are skipped, so re-run it any time to pick up what the
tracker's workflow has written since. Needed only until the collector logs these rows itself
(plan step 4); then the tracker is archived and this script goes away.

Reads TRACKER_DATABASE_URL (source) and DATABASE_URL (camanchaca's own project, the
destination) from the environment or `.env`; SRC_DATABASE_URL / DST_DATABASE_URL override them.

    scripts/sync_archive.sh                     # or: make sync-archive
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TABLES = ("forecast_obs_pairs", "ml_corrections", "station_obs_extra")
BATCH = 5000


def columns(cur, table: str) -> list[tuple[str, str]]:
    cur.execute(
        """SELECT column_name, format_type(a.atttypid, a.atttypmod)
           FROM information_schema.columns c
           JOIN pg_attribute a ON a.attrelid = c.table_name::regclass AND a.attname = c.column_name
           WHERE c.table_schema = 'public' AND c.table_name = %s
           ORDER BY c.ordinal_position""",
        (table,),
    )
    return cur.fetchall()


def constraints(cur, table: str) -> list[str]:
    """PRIMARY KEY and UNIQUE definitions, as DDL fragments."""
    cur.execute(
        """SELECT pg_get_constraintdef(oid) FROM pg_constraint
           WHERE conrelid = %s::regclass AND contype IN ('p', 'u') ORDER BY contype""",
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def main() -> int:
    src_url = os.environ.get("SRC_DATABASE_URL") or os.environ.get("TRACKER_DATABASE_URL")
    dst_url = os.environ.get("DST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not src_url or not dst_url:
        print("set TRACKER_DATABASE_URL and DATABASE_URL in .env", file=sys.stderr)
        return 2
    if src_url == dst_url:
        print("TRACKER_DATABASE_URL and DATABASE_URL are the same database", file=sys.stderr)
        return 2
    src, dst = psycopg2.connect(src_url), psycopg2.connect(dst_url)
    try:
        for table in TABLES:
            with src.cursor() as s, dst.cursor() as d:
                cols = columns(s, table)
                cons = constraints(s, table)
                parts = [f'"{name}" {typ}' for name, typ in cols] + cons
                d.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(parts)})')
                for name, typ in cols:
                    d.execute(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{name}" {typ}')
                dst.commit()
                names = ", ".join(f'"{n}"' for n, _ in cols)
                conflict = "ON CONFLICT DO NOTHING" if cons else ""
                copied = 0
                with src.cursor(name=f"sync_{table}") as reader:
                    reader.itersize = BATCH
                    reader.execute(f'SELECT {names} FROM "{table}"')
                    while True:
                        rows = reader.fetchmany(BATCH)
                        if not rows:
                            break
                        psycopg2.extras.execute_values(
                            d, f'INSERT INTO "{table}" ({names}) VALUES %s {conflict}', rows,
                            page_size=1000,
                        )
                        copied += len(rows)
                        dst.commit()
                d.execute(f'SELECT count(*) FROM "{table}"')
                total = d.fetchone()[0]
                print(f"{table:20s} read {copied:>8,}   now {total:>8,} rows")
    finally:
        src.close()
        dst.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
