from pydantic import BaseModel, Field, ConfigDict
import datetime
import requests
from typing import Any
import polars as pl


class WeatherDataIngestion(BaseModel, arbitrary_types_allowed = True, extra="forbid"):
    """Fetches raw weather data from Open-Meteo.
    Owns: API connection, raw schema validation, raw Parquet write."""
    
    historical_url: str = Field(
        default="https://archive-api.open-meteo.com/v1/archive",
        description="The API endpoint for fetching weather data."
    )
    forecast_url: str = Field(
        default="https://api.open-meteo.com/v1/forecast",
        description="The API endpoint for fetching forecast weather data."
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class IngestionConfig(BaseModel):
        
        latitude: float = Field(default=51.5085)
        longitude: float = Field(default=-0.1257)
        start_date: str = Field(default="2009-01-01", description= "Start date for weather data in YYYY-MM-DD format.")
        end_date: str = Field(default="2024-10-03", description= "End date for weather data in YYYY-MM-DD format.")
        timezone: str = Field(default="Europe/London", description= "Timezone location for the weather data, e.g., 'Europe/London'.")
        forecast_length: int = Field(default=10, description= "Number of days ahead for which to fetch forecast weather data.")
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

    settings: IngestionConfig = Field(default_factory=IngestionConfig)


    def weather_data_schema() -> dict[str, Any]:
        """Defines the imposed schema for the weather data."""
        return {
            "time": pl.Datetime,
            "temperature_2m": pl.Float32,
            "apparent_temperature": pl.Float32,
            "precipitation": pl.Float32,
            "cloud_cover": pl.Float32,
            "wind_speed_10m": pl.Float32,
            "wind_speed_100m": pl.Float32,
            "shortwave_radiation": pl.Float32,
        }

    def fetch_historical_weather_data(self) -> pl.DataFrame:
        """Fetches weather data from the API and returns it as a Polars DataFrame."""
        
        response = requests.get(self.historical_url, params=self.settings.params)
        data = response.json()
        data = pl.DataFrame(data["hourly"])
        data = data.cast(self.weather_data_schema())
        
        return data
    
    def fetch_forecast_weather_data(self, horizon_hours: int) -> pl.DataFrame:
        """Fetches forecast weather data from the API and returns it as a Polars DataFrame."""
            # Adjust the end date to be the current date plus the forecast horizon
        response = requests.get() # provide url for NWP (Numerical Weather Prediction) forecasts — published ahead of time, updated every 6-12 hours by providers like the Met Office or ECMWF.
        data = response.json()
        data = pl.DataFrame(data[''])
        data = data.cast(self.weather_data_schema())
        
        return data
        ...


# Required — tells Pydantic to resolve the forward reference to IngestionConfig
WeatherIngestion.model_rebuild()

## Application All defaults
ingestion = WeatherIngestion()

# Custom config
test_config = WeatherIngestion.IngestionConfig(
    url="http://localhost:8080/mock",
    start_date="2024-01-01",
    end_date="2024-01-02",
)
ingestion = WeatherIngestion(settings=test_config)
    
