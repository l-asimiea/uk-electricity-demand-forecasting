import polars as pl
from pydantic import BaseModel


class WeatherDataProcessor(BaseModel):
    """Processes raw weather data."""

    file_paths: list

    @staticmethod
    def unprocessed_data_schema() -> dict:
        """Schema for unprocessed data log"""
        return {
            "filepath": pl.Utf8,
            "error": pl.Utf8,
        }

    def rename_columns(df):
        """Renames columns"""
        return df.rename(
            {
                "apparent_temperature": "temperature_apparent",
                "time": "settlement_datetime",
            }
        )

    @staticmethod
    def weather_data_schema() -> dict[str, pl.DataType]:
        """Defines the imposed schema for the weather data."""
        return {
            "settlement_period_utc": pl.Datetime("us", "UTC"),
            "temperature_2m": pl.Float32,
            "apparent_temperature": pl.Float32,
            "precipitation": pl.Float32,
            "cloud_cover": pl.Float32,
            "wind_speed_10m": pl.Float32,
            "wind_speed_100m": pl.Float32,
            "shortwave_radiation": pl.Float32,
        }

    def create_settlement_period(df) -> pl.DataFrame:
        """Creates settlement period"""
        data = (
            df.with_columns(pl.col("settlement_datetime").str.strptime(pl.Datetime).alias("settlement_datetime"))
            .with_columns(
                ((pl.col("settlement_datetime").dt.hour() * 2) + 1).alias("period_start"),
                ((pl.col("settlement_datetime").dt.hour() * 2) + 2).alias("period_end"),
            )
            .with_columns(pl.concat_list(["period_start", "period_end"]).alias("period"))
            .explode("period")
            .drop(["period_start", "period_end"])
        )
        return data

    def execute(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Processes the raw data weather data"""

        processed_data = []
        unprocessed_weather_data = []

        for file in self.file_paths:
            try:
                data = pl.scan.parquet(file)

                data = self.rename_columns(data)

                data = self.create_settlement_period(data)

                processed_data.append(data)

            except Exception as e:  #
                unprocessed_weather_data.append({"filepath": file, "error": str(e)})

        processed_weather_data = pl.concat(processed_data).collect() if processed_data else pl.DataFrame()
        processed_weather_data = processed_weather_data.cast(self.output_data_schema())

        return processed_weather_data, pl.DataFrame(unprocessed_weather_data, schema=self.unprocessed_weather_data_schema())
