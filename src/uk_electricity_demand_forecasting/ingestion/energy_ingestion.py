import os
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnergyDataIngestion(BaseModel):
    """Fetches and ingests raw energy demand data"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    class IngestionConfig(BaseModel):
        large_historical_data_filepath: str | None = Field(default=None, description="File path for the large historical energy data CSV.")
        large_historical_data_use: bool = Field(default=False, description=("Whether to use the large historical energy data file."))
        chunk_output_dir: str = Field(default="data/raw/energy/chunks", description="Directory to write daily chunk CSV files into.")
        date_column: str = Field(default="settlement_date", description="Name of the date column in the historical CSV.")
        date_format: str = Field(default="%Y-%m-%d", description="Date format string for parsing the date column.")

        api_source: Literal["elexon", "entsoe"] = Field(default="elexon", description="API source to use for daily energy data ingestion.")
        elexon_base_url: str = Field(default="https://data.elexon.co.uk/bmrs/api/v1", description="Base URL for the Elexon BMRS API.")
        entsoe_base_url: str = Field(default="https://web-api.tp.entsoe.eu/api", description="Base URL for the ENTSO-E Transparency Platform API.")
        api_key: str | None = Field(
            default=None, description=("API key for the selected source. " "For Elexon: not required for public endpoints. " "For ENTSO-E: required.")
        )
        start_date: str = Field(default="2024-01-01", description="Start date for API ingestion in YYYY-MM-DD format.")
        end_date: str = Field(default="2024-10-03", description="End date for API ingestion in YYYY-MM-DD format.")
        timezone: str = Field(default="Europe/London", description="Timezone for settlement period alignment.")

        @model_validator(mode="after")
        def validate_config(self) -> "EnergyDataIngestion.IngestionConfig":
            if self.large_historical_data_use:
                if not self.large_historical_data_filepath:
                    raise ValueError("large_historical_data_filepath must be provided when large_historical_data_use is True.")
                if not Path(self.large_historical_data_filepath).exists():
                    raise ValueError(f"File not found: {self.large_historical_data_filepath}")
            if self.api_source == "entsoe" and not self.api_key:
                # Fall back to environment variable
                env_key = os.environ.get("ENTSOE_API_KEY")
                if not env_key:
                    raise ValueError("ENTSO-E requires an API key.")
                object.__setattr__(self, "api_key", env_key)
            return self

        @property
        def active_api_url(self) -> str:
            """Returns the base URL for the configured API source."""
            return self.elexon_base_url if self.api_source == "elexon" else self.entsoe_base_url

    settings: IngestionConfig = Field(default_factory=IngestionConfig)

    @staticmethod
    def energy_data_schema() -> dict[str, pl.DataType]:
        return {
            "settlement_period_utc": pl.Datetime("us", "UTC"),
            "settlement_date": pl.Date,
            "settlement_period": pl.UInt8,
            "national_demand_mw": pl.Float32,
            "transmission_system_demand_mw": pl.Float32,
            "england_wales_demand_mw": pl.Float32,
            "embedded_wind_generation_mw": pl.Float32,
            "embedded_solar_generation_mw": pl.Float32,
            "wind_capacity_mw": pl.Float32,
            "solar_capacity_mw": pl.Float32,
        }

    def split_large_historical_file(self, df) -> None:
        """Splits the large historical CSV into daily parquet files to simulate daily ingestion cadence."""

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

    def load_large_historical_file(self) -> pl.DataFrame:
        """Loads and returns the full historical CSV.

        Returns:
            pl.DataFrame: The loaded historical energy data.
        """

        if not self.settings.large_historical_data_use:
            raise RuntimeError("large_historical_data_use is False. Enable it in IngestionConfig to use this method.")
        return pl.read_csv(self.settings.large_historical_data_filepath, schema=self.energy_data_schema())

    def fetch_elexon(self) -> None:
        """Unfinished: Fetches from Elexon."""

        # data = fetch logic

        # output_path = Path(self.settings.output_dir) / f"energy_elexon_{self.settings.start_date}_{self.settings.end_date}.parquet"
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # data.write_parquet(output_path)

    def fetch_entsoe(self) -> None:
        """Unfinished: Fetches energy data from ENTSO-E."""

        # data = fetch logic

        # output_path = Path(self.settings.output_dir) / f"energy_entsoe_{self.settings.start_date}_{self.settings.end_date}.parquet"
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # data.write_parquet(output_path)

    def fetch_api(self) -> None:
        """Routes to the correct API fetcher based on configured source."""
        if self.settings.api_source == "elexon":
            self.fetch_elexon()
        else:
            self.fetch_entsoe()

    def execute(self) -> None:
        """Runs energy data ingestion using either historical files or fresh api call."""

        if self.settings.large_historical_data_use:
            df = self.load_large_historical_file()
            self.split_large_historical_file(df)
        else:
            self.fetch_api()


EnergyDataIngestion.model_rebuild()


# # use
# # File-based
# ingestion = EnergyDataIngestion(
#     settings=EnergyDataIngestion.IngestionConfig(
#         large_historical_data_use=True,
#         large_historical_data_filepath="data/raw/energy/historic_2009_2024.csv",
#         chunk_output_dir="data/raw/energy/chunks",
#         output_dir="data/raw/energy",
#     )
# )
# ingestion.execute()

# # API-based
# ingestion = EnergyDataIngestion(
#     settings=EnergyDataIngestion.IngestionConfig(
#         api_source="elexon",
#         start_date="2024-09-01",
#         end_date="2024-10-03",
#         output_dir="data/raw/energy",
#     )
# )
# ingestion.execute()
