from cellview.db.db import CellViewDB
from cellview.exporters import export_pandas_df


def export_plate():
    db = CellViewDB()
    conn = db.connect()
    df, _ = export_pandas_df(1, conn)
    print(df.head())
    print(df.columns)
    print(df.classifier.unique())
    print(df.c604.unique())
    print(df.palb.unique())
    print(df.clone.unique())


if __name__ == "__main__":
    export_plate()
