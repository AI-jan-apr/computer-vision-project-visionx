"""
Manage VisionX SQLite ground truth (plate PK + test image mapping).

  python db_cli.py init
  python db_cli.py add-plate --final "1234 ABC" --digits 1234 --letters ABC --brand Toyota --model Camry
  python db_cli.py link --image mycar.jpg --plate "1234 ABC"
  python db_cli.py show --image mycar.jpg
  python db_cli.py import-excel "dataset.xlsx"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.config import DEFAULT_DB_PATH
from utils.database import (
    get_expected_for_basename,
    get_ground_truth_by_plate_final,
    init_db,
    link_test_image,
    upsert_ground_truth,
)
from utils.excel_import import import_ground_truth_xlsx


def main() -> None:
    p = argparse.ArgumentParser(
        description="VisionX ground-truth database",
        epilog="Tip: pass --db before the subcommand, e.g.  python db_cli.py --db data/x.db import-excel file.xlsx",
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite file path")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create tables if missing")

    ap = sub.add_parser("add-plate", help="Insert/update ground truth (plate_final = PRIMARY KEY)")
    ap.add_argument("--final", required=True, help='Full plate as stored key, e.g. "1234 ABC"')
    ap.add_argument("--digits", required=True)
    ap.add_argument("--letters", required=True)
    ap.add_argument("--brand", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--color", default=None, help="Expected body color (e.g. White)")

    im = sub.add_parser(
        "import-excel",
        help="Load rows from Excel (License Plate, Brand, Model, Color) into SQLite",
    )
    im.add_argument("xlsx", type=Path, help="Path to .xlsx file")
    im.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index (default: 0)",
    )

    lk = sub.add_parser("link", help="Map test_images filename -> expected plate_final")
    lk.add_argument("--image", required=True, help="Basename only, e.g. car01.jpg")
    lk.add_argument("--plate", required=True, help="Must match an existing plate_final key")

    sh = sub.add_parser("show", help="Print expected row for an image basename")
    sh.add_argument("--image", required=True)

    sp = sub.add_parser("show-plate", help="Print registry row by plate key (e.g. \"6304 LAJ\")")
    sp.add_argument("--plate", required=True)

    args = p.parse_args()
    db = args.db

    if args.cmd == "init":
        init_db(db)
        print("OK:", db.resolve())
        return

    if args.cmd == "add-plate":
        key = upsert_ground_truth(
            args.final,
            args.digits,
            args.letters,
            expected_brand=args.brand,
            expected_model=args.model,
            expected_color=args.color,
            db_path=db,
        )
        print("Upserted plate_ground_truth:", key)
        return

    if args.cmd == "import-excel":
        n = import_ground_truth_xlsx(args.xlsx, sheet_name=args.sheet, db_path=db)
        print(f"Imported {n} rows from", args.xlsx.resolve())
        return

    if args.cmd == "link":
        link_test_image(args.image, args.plate, db_path=db)
        print("Linked", Path(args.image).name, "->", args.plate)
        return

    if args.cmd == "show":
        row = get_expected_for_basename(args.image, db_path=db)
        if row is None:
            print("No mapping for", Path(args.image).name, file=sys.stderr)
            sys.exit(1)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return

    if args.cmd == "show-plate":
        row = get_ground_truth_by_plate_final(args.plate, db_path=db)
        if row is None:
            print("No row for plate:", args.plate, file=sys.stderr)
            sys.exit(1)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
