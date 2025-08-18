import pandas as pd
from typing import Sequence, Any, Callable


class Table:
    def __init__(
        self,
        data: pd.DataFrame,
        blocks: Sequence[Sequence[Any]],
        matcher: Callable[[pd.Series, pd.Series], bool],
        alias: str,
        id_column: str,
    ):
        self.data = data
        self.matcher = matcher
        self.data = data.rename(columns=lambda x: f"{alias}_{x}")
        self.alias = alias
        self.id_column = f"{alias}_{id_column}"
        self.blocks = [
            self.data[self.data[self.id_column].isin(block)] for block in blocks
        ]
