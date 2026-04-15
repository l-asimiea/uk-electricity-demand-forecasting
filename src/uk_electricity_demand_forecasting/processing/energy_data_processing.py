import polars as pl
from pydantic import BaseModel


class EnergyDataProcessor(BaseModel):
    """This handles the processing of energy data."""

    file_paths: list

    def columns_of_interest(df) -> list:
        """Selects the columns of interest"""
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
                "settlement_date": "settlement_datetime",
                "tsd": "transmission_system_demand",
                "nd": "national_demand",
                "embedded_solar_generation": "solar_embedded_generation",
                "embedded_wind_generation": "wind_embedded_generation",
                "embedded_wind_capacity": "wind_embedded_capacity",
                "embedded_solar_capacity": "solar_embedded_capacity",
            }
        )

    @staticmethod
    def output_data_schema() -> dict:
        """Applies schema to the energy data"""
        return {
            "settlement_datetime": pl.Datetime,
            "settlement_period": pl.Int8,
            "transmission_demand": pl.Int16,
            "national_demand": pl.Int16,
            "england_wales_demand": pl.Int16,
            "solar_embedded_generation": pl.Int8,
            "wind_embedded_generation": pl.Int8,
            "is_holiday": pl.Boolean,
        }

    @staticmethod
    def unprocessed_data_schema() -> dict:
        """Schema for unprocessed data log"""
        return {
            "filepath": pl.Utf8,
            "error": pl.Utf8,
        }

    def apply_data_quality(df) -> pl.DataFrame:
        """Applies data quality steps to clean the data.
        Args:
        df: pl.DataFrame - the input dataframe

        Returns:
        pl.DataFrame - A dataframe with improved quality
        """

        # Remove outliers on tsd but extremes from IQR can be used
        df = df.filter(pl.col("transmission_demand") > 10000)

        # enrich date to datetime stamp
        df = (
            df.with_columns(
                pl.col("settlement_date").str.to_date("%Y-%m-%d").alias("settlement_date"),
                pl.duration(minutes=((pl.col("settlement_period") - 1) * 30)).cast(pl.Time).alias("period_time"),
            )
            .with_columns(
                (pl.col("settlement_date").cast(pl.Datetime) + pl.col("period_time")).alias("settlement_datetime"),
            )
            .drop(["settlement_date", "period_time"])
        )

        # Keep only periods less than 50
        df = df.filter(pl.col("settlement_period") < 51)

        return df

    def execute(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Processes the raw energy data"""

        processed_data = []
        unprocessed_data = []

        for file in self.file_paths:
            try:
                data = pl.scan.parquet(file)

                data = data.select(self.columns_of_interest(data))

                data = self.rename_columns(data)

                data = self.apply_data_quality(data)

                processed_data.append(data)

            except Exception as e:  #
                unprocessed_data.append({"filepath": file, "error": str(e)})

        processed_energy_data = pl.concat(processed_data).collect() if processed_data else pl.DataFrame()
        processed_energy_data = processed_energy_data.cast(self.output_data_schema())

        return processed_energy_data, pl.DataFrame(unprocessed_data, schema=self.unprocessed_data_schema())
