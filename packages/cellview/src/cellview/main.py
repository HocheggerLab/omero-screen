from cellview.db.clean_up import clean_up_db
from cellview.db.db import CellViewDB
from cellview.db.display import display_plate_summary, display_projects
from cellview.importers import import_from_csv
from cellview.utils.state import CellViewState

from .cli import parse_args


def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    db = CellViewDB(args.db)
    conn = db.connect()
    if args.csv:
        state = CellViewState.get_instance(args)
        import_from_csv(db, state)

    if args.clean:
        clean_up_db(db, db.connect())
    if args.plate:
        display_plate_summary(args.plate, conn)
    if args.projects:
        display_projects(conn)
    conn.close()


if __name__ == "__main__":
    main()
