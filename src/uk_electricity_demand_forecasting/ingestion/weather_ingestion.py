from pathlib import Path
from typing import Any

import polars as pl
import requests
from pydantic import BaseModel, ConfigDict, Field


class WeatherDataIngestion(BaseModel):
    """Fetches raw weather data from Open-Meteo."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    class IngestionConfig(BaseModel):
        historical_url: str = Field(default="https://archive-api.open-meteo.com/v1/archive", description="API endpoint for historical weather data.")
        forecast_url: str = Field(default="https://api.open-meteo.com/v1/forecast", description="API endpoint for forecast weather data.")

        # ── File-based historical ingestion ──────────────────────────────
        large_historical_data_filepath: str | None = Field(default=None, description="File path for the large historical weather data CSV.")
        large_historical_data_use: bool = Field(
            default=False,
            description=(
                "Whether to use the large historical weather data file. " "Set to True to enable splitting and ingestion of the large file."
            ),
        )
        chunk_output_dir: str = Field(default="data/raw/weather/chunks", description="Directory to write daily chunk CSV files into.")

        latitude: float = Field(default=51.5085)
        longitude: float = Field(default=-0.1257)
        start_date: str = Field(default="2009-01-01", description="Start date in YYYY-MM-DD format.")
        end_date: str = Field(default="2024-10-03", description="End date in YYYY-MM-DD format.")
        timezone: str = Field(default="Europe/London", description="Timezone for the weather data.")
        forecast_length: int = Field(default=10, description="Number of days ahead for forecast data.")
        hourly_variables: list[str] = Field(
            default=[
                "temperature_2m",
                "apparent_temperature",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "wind_speed_100m",
                "shortwave_radiation",
            ]
        )

        @property
        def params(self) -> dict[str, Any]:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "timezone": self.timezone,
                "hourly": self.hourly_variables,
            }

        @property
        def forecast_params(self) -> dict[str, Any]:
            return {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "forecast_days": self.forecast_length,
                "timezone": self.timezone,
                "hourly": self.hourly_variables,
            }

    settings: IngestionConfig = Field(default_factory=IngestionConfig)

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

    def load_large_historical_weather_file(self) -> pl.DataFrame:
        """Loads and returns the full historical weather CSV as a Polars DataFrame."""

        if not self.settings.large_historical_data_use:
            raise RuntimeError("large_historical_data_use is False. Enable it in IngestionConfig to use this method.")

        return pl.read_csv(self.settings.large_historical_data_filepath, try_parse_dates=False)

    def split_large_historical_weather_file(self, df) -> None:
        """Splits the large historical weather CSV into daily parquet files to simulate daily ingestion cadence."""

        if not self.settings.large_historical_data_use:
            raise RuntimeError("large_historical_data_use is False. Enable it in IngestionConfig to use this method.")

        output_dir = Path(self.settings.chunk_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = df.with_columns(pl.col(self.settings.date_column).str.to_date(format=self.settings.date_format).alias(self.settings.date_column))

        date_groups = data.partition_by(self.settings.date_column, as_dict=True)

        for date_key, date_group in date_groups.items():
            output_path = output_dir / f"energy_{date_key}.parquet"
            date_group.write_parquet(output_path)

        print(f"Written {len(date_groups)} daily files to {output_dir}")

    def fetch_historical_weather_data(self) -> pl.DataFrame:
        """Fetches weather data from the API and returns it as a Polars DataFrame."""

        response = requests.get(self.historical_url, params=self.settings.params)
        data = response.json()
        data = pl.DataFrame(data["hourly"])
        data = data.cast(self.weather_data_schema())

        output_path = Path(self.settings.output_dir) / f"weather_data_{self.settings.start_date}_{self.settings.end_date}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(output_path)

    def fetch_forecast_weather_data(self, horizon_hours: int) -> pl.DataFrame:
        """Unfinished: Fetches forecast weather data from the API and returns it as a Polars DataFrame."""
        # Adjust the end date to be the current date plus the forecast horizon
        response = requests.get()  # provide url for Numerical Weather Prediction) forecasts
        data = response.json()
        data = pl.DataFrame(data[""])
        data = data.cast(self.weather_data_schema())

        return data

    def execute(self) -> None:
        """Processes the ingestion of weather data using either historical files or fresh api call."""

        if self.settings.large_historical_data_use:
            df = self.load_large_historical_weather_file()
            self.split_large_historical_weather_file(df)
        else:
            self.fetch_historical_weather_data()


# This tells Pydantic to resolve the forward reference to IngestionConfig
WeatherDataIngestion.model_rebuild()

## Application All defaults
ingestion = WeatherDataIngestion()

# Custom config
test_config = WeatherDataIngestion.IngestionConfig(
    url="http://localhost:8080/mock",
    start_date="2024-01-01",
    end_date="2024-01-02",
)
ingestion = WeatherDataIngestion(settings=test_config)
