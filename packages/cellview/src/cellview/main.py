from cellview.db.clean_up import clean_up_db, del_measurements_by_plate_id
from cellview.db.db import CellViewDB
from cellview.db.display import display_plate_summary, display_projects
from cellview.importers import import_data
from cellview.utils.state import CellViewState

from .cli import parse_args


def main() -> None:
    """Main entry point for the application."""
    args = parse_args()
    db = CellViewDB(args.db)
    conn = db.connect()
    if args.csv:
        state = CellViewState.get_instance(args)
        import_data(db, state)
    if args.plate_id:
        state = CellViewState.get_instance(args)
        import_data(db, state)
    if args.clean:
        clean_up_db(db, conn)
    if args.plate:
        display_plate_summary(args.plate, conn)
    if args.projects:
        display_projects(conn)
    if args.delete_plate:
        del_measurements_by_plate_id(db, conn, args.delete_plate)

    conn.close()


if __name__ == "__main__":
    main()
