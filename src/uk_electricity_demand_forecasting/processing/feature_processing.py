"""
This script contains all functions relevant for featuring engineering operations
implemented on the dataset prior to or during model development.

Input:
    df:         the dataframe containing source variables for featuring engineering

Output:
    data_uk:    the dataframe including created features

"""

# Imported Libraries
import os

import numpy as np
import plotly.express as px
import plotly.io as pio
import polars as pl
from pydantic import BaseModel

from src.visualisation.plot_utils import plotly_user_standard_settings

plotly_user_standard_settings(pio, px)

# path to saving plots
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
save_path = os.path.join(project_root, "reports/figures/")


class FeaturesProcessor(BaseModel):
    input_energy_data: pl.DataFrame
    input_energy_data: pl.DataFrame

    @staticmethod
    def temporal_features_schema() -> dict:
        """Applies the schema for the temporal features."""
        return {
            "hours": pl.Int8,
            "day_of_week": pl.Int8,
            "day_of_month": pl.Int8,
            "day_of_year": pl.Int16,
            "week": pl.Int8,
            "month": pl.Int8,
            "quarter": pl.Int8,
            "year": pl.Int16,
            "is_weekend": pl.Int8,
        }

    @staticmethod
    def output_schema() -> dict:
        """Applies the schema to teh final features dataset."""
        return {
            "settlement_datetime": pl.Datetime,
            "settlement_period": pl.Int8,
            "transmission_demand": pl.Float32,
            "is_holiday": pl.Int8,
            "hours": pl.Int8,
            "day_of_week": pl.Int8,
            "day_of_month": pl.Int8,
            "day_of_year": pl.Int16,
            "week": pl.Int8,
            "month": pl.Int8,
            "quarter": pl.Int8,
            "year": pl.Int16,
            "is_weekend": pl.Int8,
            "hour_sin": pl.Float32,
            "hour_cos": pl.Float32,
            "day_sin": pl.Float32,
            "day_cos": pl.Float32,
            "month_sin": pl.Float32,
            "month_cos": pl.Float32,
            "lag_30min": pl.Int16,
            "lag_1hour": pl.Int16,
            "lag_1day": pl.Int16,
            "lag_1week": pl.Int16,
            "lag_1year": pl.Int16,
            "lag_2year": pl.Int16,
            "rolling_mean_1day": pl.Float32,
        }

    def create_temporal_features(self, df) -> pl.DataFrame:
        """Creates time features as well as lag features lag1, 2, 3 and 5
        Args:
        df:         data as a pandas dataframe

        Return:
        data:    the feature engineered data
        """

        data = df.input_data.with_columns(
            pl.col("settlement_datetime").dt.hour().alias("hours"),
            pl.col("settlement_datetime").dt.weekday().alias("day_of_week"),
            pl.col("settlement_datetime").dt.day().alias("day_of_month"),
            pl.col("settlement_datetime").dt.ordinal_day().alias("day_of_year"),
            pl.col("settlement_datetime").dt.week().alias("week"),
            pl.col("settlement_datetime").dt.month().alias("month"),
            pl.col("settlement_datetime").dt.quarter().alias("quarter"),
            pl.col("settlement_datetime").dt.year().alias("year"),
        )
        data = data.with_columns(pl.when(pl.col("day-of_week") >= 5).then(1).otherwise(0).cast(pl.Int8).alias("is_weekend"))

        return data.cast(self.temporal_features_schema())

    def create_encoded_features(df) -> pl.DataFrame:
        """Creates encoded features from the temporal features."""
        data = df.with_columns(
            hour_sin=np.sin(2 * np.pi * pl.col("hours") / 24),
            hour_cos=np.cos(2 * np.pi * pl.col("hours") / 24),
            day_sin=np.sin(2 * np.pi * pl.col("day_of_week") / 7),
            day_cos=np.cos(2 * np.pi * pl.col("day_of_week") / 7),
            month_sin=np.sin(2 * np.pi * pl.col("month") / 12),
            month_cos=np.cos(2 * np.pi * pl.col("month") / 12),
        )
        return data

    def create_lag_features(df) -> pl.DataFrame:
        """Add lags to the dataset."""
        # Create lag features
        lags_df = df.with_columns(
            pl.col("transmission_demand").shift(1).alias("lag_30min"),
            pl.col("transmission_demand").shift(2).alias("lag_1hour"),
            pl.col("transmission_demand").shift(48).alias("lag_1day"),
            pl.col("transmission_demand").shift(336).alias("lag_1week"),
            pl.col("transmission_demand").shift(52 * 336).alias("lag_1year"),
            pl.col("transmission_demand").shift(2 * 52 * 336).alias("lag_2year"),
            pl.col("transmission_demand").shift(1).rolling(48).mean().alias("rolling_mean_1day"),  # to avoid data leakage
        )

        return lags_df

    def execute(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Processes the features from the energy and weather data"""

        if not self.input_energy_data or not self.input_weather_data:
            raise ValueError("Both energy and weather data must be provided.")

        # Join energy and weather data
        data = self.input_energy_data.select(["settlement_datetime", "settlement_period", "transmission_demand", "is_holiday"]).join(
            self.input_weather_data, on="settlement_datetime", how="left"
        )

        data = data.sort("settlement_datetime")

        data_w_temporal_features = self.create_temporal_features(data)
        data_w_encoded_features = self.create_encoded_features(data_w_temporal_features)
        data_w_lag_features = self.create_lag_features(data_w_encoded_features)
        features_data = data_w_lag_features.cast(self.output_schema())

        return features_data.sort("settlement_datetime")
