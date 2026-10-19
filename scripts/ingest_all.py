"""Script to ingest the complete STAGE dataset pack into SQLite."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage_agent.config import DATA_PACK_DIR, DB_PATH
from stage_agent.db import (
    init_db,
    ingest_catalogue,
    ingest_episodes,
    ingest_events,
    ingest_profiles,
)


def main():
    parser = argparse.ArgumentParser(description="Ingest the STAGE dataset into SQLite.")
    parser.add_argument(
        "--pack",
        type=Path,
        default=DATA_PACK_DIR,
        help="Path to the one_user_one_agent_pack directory.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to the output SQLite database file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for event ingestion.",
    )
    args = parser.parse_args()

    pack_dir = args.pack
    db_path = args.db

    print(f"==================================================")
    print(f"STAGE One User, One Agent - Dataset Ingestion")
    print(f"Data pack: {pack_dir}")
    print(f"Database:  {db_path}")
    print(f"==================================================")

    start_total = time.perf_counter()
    conn = init_db(db_path)

    # 1. Ingest Profiles Snapshot
    prof_file = pack_dir / "data" / "profiles_snapshot.csv"
    print(f"\n[1/4] Ingesting profiles from {prof_file} ...")
    t0 = time.perf_counter()
    res_prof = ingest_profiles(prof_file, conn)
    t1 = time.perf_counter()
    print(f"      Read:      {res_prof.records_read:,}")
    print(f"      Inserted:  {res_prof.records_inserted:,} (Authoritative unique profiles)")
    print(f"      Updated:   {res_prof.records_updated:,} (Newer updated_at records)")
    print(f"      Duplicates:{res_prof.duplicates_skipped:,} (Older updated_at records)")
    print(f"      Malformed: {res_prof.malformed_count:,}")
    print(f"      Elapsed:   {t1 - t0:.2f}s")

    # 2. Ingest Catalogue
    cat_file = pack_dir / "data" / "catalogue.csv"
    print(f"\n[2/4] Ingesting catalogue from {cat_file} ...")
    t0 = time.perf_counter()
    res_cat = ingest_catalogue(cat_file, conn)
    t1 = time.perf_counter()
    print(f"      Read:      {res_cat.records_read:,}")
    print(f"      Inserted:  {res_cat.records_inserted:,}")
    print(f"      Updated:   {res_cat.records_updated:,}")
    print(f"      Malformed: {res_cat.malformed_count:,}")
    print(f"      Elapsed:   {t1 - t0:.2f}s")

    # 3. Ingest Episodes
    ep_file = pack_dir / "data" / "episodes.csv"
    print(f"\n[3/4] Ingesting episodes from {ep_file} ...")
    t0 = time.perf_counter()
    res_ep = ingest_episodes(ep_file, conn)
    t1 = time.perf_counter()
    print(f"      Read:      {res_ep.records_read:,}")
    print(f"      Inserted:  {res_ep.records_inserted:,}")
    print(f"      Updated:   {res_ep.records_updated:,}")
    print(f"      Malformed: {res_ep.malformed_count:,}")
    print(f"      Elapsed:   {t1 - t0:.2f}s")

    # 4. Ingest Events
    events_file = pack_dir / "data" / "events.jsonl"
    print(f"\n[4/4] Ingesting events from {events_file} (batch size {args.batch_size}) ...")
    t0 = time.perf_counter()
    res_ev = ingest_events(events_file, conn, batch_size=args.batch_size)
    t1 = time.perf_counter()
    print(f"      Read:      {res_ev.records_read:,}")
    print(f"      Inserted:  {res_ev.records_inserted:,}")
    print(f"      Duplicates:{res_ev.duplicates_skipped:,}")
    print(f"      Malformed: {res_ev.malformed_count:,}")
    print(f"      Elapsed:   {t1 - t0:.2f}s")

    # Query summary counts
    print(f"\n--------------------------------------------------")
    print(f"Database Verification:")
    ev_total = conn.execute("SELECT count(*) FROM events;").fetchone()[0]
    prof_auth = conn.execute("SELECT count(*) FROM profiles;").fetchone()[0]
    prof_raw = conn.execute("SELECT count(*) FROM profiles_raw;").fetchone()[0]
    cat_total = conn.execute("SELECT count(*) FROM catalogue;").fetchone()[0]
    ep_total = conn.execute("SELECT count(*) FROM episodes;").fetchone()[0]
    off_total = conn.execute("SELECT count(*) FROM events WHERE is_offline_buffered = 1;").fetchone()[0]

    print(f"• Total events in database:           {ev_total:,} (Offline buffered: {off_total:,})")
    print(f"• Unique authoritative profiles:      {prof_auth:,}")
    print(f"• Raw profile snapshots preserved:    {prof_raw:,}")
    print(f"• Total catalogue titles:             {cat_total:,}")
    print(f"• Total episodes:                     {ep_total:,}")
    print(f"Total time: {time.perf_counter() - start_total:.2f}s")
    print(f"==================================================")
    conn.close()


if __name__ == "__main__":
    main()
