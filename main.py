import polars as pl

pl.scan_csv("final_dataset.csv").sink_parquet("output.parquet")
