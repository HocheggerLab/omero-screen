from cellview.exporters import export_pandas_df


def test_export_plate():
    df = export_pandas_df(1)
    print(df.head())
    print(df.columns)
    print(df.classifier.unique())
    print(df.c604.unique())
    print(df.palb.unique())
    print(df.clone.unique())
