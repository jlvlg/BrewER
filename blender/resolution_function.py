from abc import ABC, abstractmethod
import pandas as pd
from typing import Any
from .util import SqlOrderBy


class ResolutionFunction(ABC):
    @staticmethod
    @abstractmethod
    def resolve(series: pd.Series) -> Any: ...

    @staticmethod
    def is_discordant(order: SqlOrderBy) -> bool:
        return False

    @staticmethod
    def extract_seed_records_eq(column: str, value: str) -> str:
        return f"{column} == {value}"

    @staticmethod
    def extract_seed_records_neq(column: str, value: str) -> str:
        return f"{column} != {value}"

    @staticmethod
    def extract_seed_records_gt(column: str, value: str) -> str:
        return f"{column} > {value}"

    @staticmethod
    def extract_seed_records_gte(column: str, value: str) -> str:
        return f"{column} >= {value}"

    @staticmethod
    def extract_seed_records_lt(column: str, value: str) -> str:
        return f"{column} < {value}"

    @staticmethod
    def extract_seed_records_lte(column: str, value: str) -> str:
        return f"{column} <= {value}"

    @staticmethod
    def extract_seed_records_like(column: str, value: str) -> str:
        return f"{column}.str.contains(na=False, pat={value})"

    @staticmethod
    def extract_seed_records_ilike(column: str, value: str) -> str:
        return f"{column}.str.contains(na=False, case=False, pat={value})"


class VOTE(ResolutionFunction):
    @staticmethod
    def resolve(series: pd.Series):
        return series.mode()[0]


class MIN(ResolutionFunction):
    @staticmethod
    def resolve(series: pd.Series):
        return series.min()

    @staticmethod
    def is_discordant(order: SqlOrderBy) -> bool:
        return order == SqlOrderBy.DESC


class MAX(ResolutionFunction):
    @staticmethod
    def resolve(series: pd.Series):
        return series.max()

    @staticmethod
    def is_discordant(order: SqlOrderBy) -> bool:
        return order == SqlOrderBy.ASC


class RANDOM(ResolutionFunction):
    @staticmethod
    def resolve(series: pd.Series):
        return series.sample(1)[0]


class AVG(ResolutionFunction):
    @staticmethod
    def resolve(series: pd.Series):
        return series.mean()

    @staticmethod
    def extract_seed_records_eq(column, value):
        return f"{column}.min() < {value} < {column}.max()"

    @staticmethod
    def extract_seed_records_neq(column, value):
        return "True"

    @staticmethod
    def extract_seed_records_gte(column, value):
        return f"{column}.min() < {value} < {column}.max()"

    @staticmethod
    def extract_seed_records_lte(column, value):
        return f"{column}.min() < {value} < {column}.max()"

    @staticmethod
    def extract_seed_records_like(column, value):
        raise TypeError("AVG cannot be applied to string values")

    @staticmethod
    def extract_seed_records_ilike(column, value):
        raise TypeError("AVG cannot be applied to string values")
