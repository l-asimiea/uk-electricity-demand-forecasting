from pydantic import BaseModel, Field, ConfigDict
import datetime
import requests
from typing import Any
import polars as pl


class EnergyDataIngestion(BaseModel, arbitrary_types_allowed = True, extra="forbid"):