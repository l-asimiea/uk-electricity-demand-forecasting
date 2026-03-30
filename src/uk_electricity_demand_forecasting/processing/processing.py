import polars as pl
from pydantic import BaseModel


class DataProcessor(BaseModel):
    """This handles the processing of the energy data."""

    file_paths: list  # list of 1 or more filepaths as strings

    def columns_of_interest(df) -> list:
        """Select the columns of interest"""
        return [
            columns
            for columns in df.columns
            if (
                columns.startswith("settlement_")
                or columns.startswith("nd")
                or columns.startswith("tsd")
                or columns.startswith("england_")
                or columns.startswith("embedded_")
                or columns.endswith("_holiday")
            )
        ]

    def rename_columns(df) -> pl.DataFrame:
        """Renames columns"""
        return df.rename(
            {
                "settlement_date": "date",
                "settlement_period": "period",
                "tsd": "transmission_demand",
                "nd": "national_demand",
                "england_whales_demand": "gb_eng_and_wls",
                "embedded_solar_generation": "embedded_solar_gen",
                "embedded_wind_generation": "embedded_wind_gen",
            }
        )

    def input_data_schema(df) -> dict:
        """Applies schema to telemetry"""
        return {
            "date": pl.Datetime,
            "period": pl.Int8,
            "transmission_demand": pl.Int16,
            "national_demand": pl.Int16,
            "gb_eng_and_wls": pl.Int16,
            "embedded_solar_gen": pl.Int8,
            "embedded_wind_gen": pl.Int8,
            "is_holiday": pl.Boolean,
        }

    def apply_data_quality(df) -> pl.DataFrame:
        """Applies data quality steps to further clean the data.
        Args:
        df: pl.DataFrame - the input dataframe

        Returns:
        pl.DataFrame - A dataframe with improved quality

        """

        # Remove outliers on tsd but extremes from IQR can be used
        df = df.filter(pl.col("tsd") > 10000)

        # enrich date to datetime stamp
        df = df.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d").alias("date_parsed"),
            pl.duration(minutes=((pl.col("period") - 1) * 30))
            .cast(pl.Time)
            .alias("period_time"),
        ).with_columns(
            (pl.col("date_parsed").cast(pl.Datetime) + pl.col("period_time")).alias(
                "timestamp"
            ),
            pl.col("period_time").dt.hour.alias("hours"),
            pl.col("period_time").dt.minute.alias("minutes"),
        )

        # Remove instances wherein there is more than 2 datapoints per hour
        df = (
            df.with_columns(
                pl.col("timestamp").dt.date().alias("day"),
                # pl.col("timestamp").dt.minute().alias("minute"),
            )
            # Keep only rows landing on :00 or :30
            .filter(pl.col("minute").is_in([0, 30]))
            # Within each day, deduplicate on the exact timestamp
            .unique(subset=["timestamp"], keep="first")
            # Verify each day has exactly 48 periods
            .with_columns(
                pl.col("timestamp").count().over("day").alias("periods_per_day")
            )
            # .drop(["minute"])
        )

        return df

    def execute(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """This is the entry point to the class and it executes teh steps required to process the raw data"""
        telemetry = pl.DataFrame()
        unprocessed_files = pl.DataFrame()

        try:
            for file in self.file_paths:
                # reads data from data file
                data = pl.read.csv(file)

                # Columns selection
                data = data.select(self.columns_of_interest(data))

                # rename columns
                data = self.rename_columns(data)

                # apply data quality check
                data = self.apply_data_quality(data)

                # Save dataframe
                if telemetry.height == 0:
                    telemetry = data
                else:
                    telemetry = telemetry.vstack(telemetry)
        except Exception as e:  #
            unprocessed_files = pl.DataFrame({"filepath": file})

            return telemetry, unprocessed_files
