from .table import Table
from .resolution_function import ResolutionFunction


class Column:
    def __init__(
        self,
        name: str,
        resolution_function: type[ResolutionFunction],
        table: Table,
    ):
        self.name = name
        self.resolution_function = resolution_function
        self.table = table
