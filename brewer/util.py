from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import total_ordering
import math
from typing import Iterable, Any, overload


class SqlOperation(StrEnum):
    SELECT = auto()
    UNKNOWN = auto()


class SqlOrderBy(StrEnum):
    ASC = auto()
    DESC = auto()


class UnorderedTupleSet:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, iterable: Iterable[tuple]): ...

    def __init__(self, iterable=()):
        self._data = set(tuple(sorted(seq)) for seq in iterable)

    def add(self, element: tuple[Any, ...]) -> None:
        self._data.add(tuple(sorted(element)))

    def __contains__(self, element: tuple[Any, ...]) -> bool:
        return tuple(sorted(element)) in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._data.__repr__()


@dataclass()
@total_ordering
class PriorityItem:
    priority: Any
    item: Any
    reverse: bool = field(default=False)
    solved: bool = field(default=False)

    def __eq__(self, other) -> bool:
        if not hasattr(other, "priority"):
            return NotImplemented

        return self.priority == other.priority

    def __lt__(self, other) -> bool:
        if not hasattr(other, "priority"):
            return NotImplemented

        if math.isnan(other.priority):
            return True
        if math.isnan(self.priority):
            return False

        if self.reverse:
            return self.priority >= other.priority
        return self.priority < other.priority
