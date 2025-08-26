from .table import Table
from .column import Column
from .grammar import (
    query_where_parser,
    query_on_parser,
    QueryWhereTransformer,
    QueryOnTransformer,
)
from .util import (
    UnorderedTupleSet,
    SqlOperation,
    SqlOrderBy,
    PriorityItem,
)
from typing import Callable, Any, Optional, Sequence
import pandas as pd
from .resolution_function import ResolutionFunction
import random
import heapq
from collections import deque
import inspect

type ListenerType = Callable


class BrewER:
    def __init__(self):
        self.listeners: list[ListenerType] = []
        self.op = SqlOperation.UNKNOWN
        self.tables: dict[Table, list[Column]] = {}
        self.conditions: dict[Table, Any] = {}
        self.seed_records: dict[Table, pd.DataFrame] = {}
        self.order_by: tuple[Column, SqlOrderBy]
        self.did_where_change = False
        self.matches: dict[Table, UnorderedTupleSet] = {}
        self.not_matches: dict[Table, UnorderedTupleSet] = {}
        self.on: dict[str, list[str]] = {}
        self.top_k = None

    def subscribe(self, *listeners: ListenerType):
        self.listeners.extend(listeners)
        return self

    def from_table(self, table: Table):
        self.tables[table] = []
        self.seed_records[table] = table.data
        self.matches[table] = UnorderedTupleSet()
        self.not_matches[table] = UnorderedTupleSet()
        return self

    def __join(self, table: Table, on: str):
        raise NotImplementedError()
        self.tables[table] = []
        self.matches[table] = UnorderedTupleSet()
        self.not_matches[table] = UnorderedTupleSet()
        self.seed_records[table] = table.data
        tree = query_on_parser.parse(on)
        columns = QueryOnTransformer().transform(tree)
        for k, v in columns.items():
            self.on.setdefault(k, []).append(v)

        return self

    def select(
        self,
        *columns: tuple[str, type[ResolutionFunction]],
        order_by: Optional[tuple[str, SqlOrderBy]] = None,
        top_k: Optional[int] = None,
    ):
        if not len(columns):
            raise IndexError("You must select at least one column")
        self.op = SqlOperation.SELECT
        self.top_k = top_k

        table_aliases = {table.alias: table for table in self.tables}

        for column in columns:
            table_alias, column_name = column[0].split(".")
            table = table_aliases.get(table_alias, None)

            if table is None:
                raise KeyError("Table not loaded")

            self.tables[table].append(
                Column(f"{table_alias}_{column_name}", column[1], table)
            )

        all_columns = [
            column for _, columns in self.tables.items() for column in columns
        ]

        if order_by is None:
            self.order_by = (random.choice(all_columns), SqlOrderBy.ASC)
        else:
            if order_by[0] not in [column[0] for column in columns]:
                raise IndexError("Sorting by unknown column")
            self.order_by = (
                [
                    column
                    for column in all_columns
                    if column.name == order_by[0].replace(".", "_")
                ][0],
                order_by[1],
            )

        return self

    def where(self, conditions: str):
        self.did_where_change = True
        tree = query_where_parser.parse(conditions)

        for table in self.tables:
            self.conditions[table] = tree

        return self

    def extract_seed_records(self):
        for table in self.tables:
            query = QueryWhereTransformer(
                columns=self.tables[table],
                extract_seed_records=True,
            ).transform(self.conditions[table])
            if query is not None:
                seeds = table.data.query(query)
            else:
                seeds = table.data
            self.seed_records[table] = seeds

    def run(self, reset_matches=False):
        if self.op == SqlOperation.SELECT:
            return self.run_select(reset_matches)

    def emit(self, entity: pd.Series, cluster: set[str], comparisons: int):
        for listener in self.listeners:
            params = len(inspect.signature(listener).parameters)
            if params == 1:
                listener(entity)
            elif params == 2:
                listener(entity, cluster)
            else:
                listener(entity, cluster, comparisons)

    def run_select(self, reset_matches):
        if not len(self.tables):
            raise IndexError("You must load at least one table")
        if self.did_where_change:
            self.extract_seed_records()
        table = self.order_by[0].table
        analyzed = set()
        queue = [
            PriorityItem(
                priority=row[self.order_by[0].name],
                item=[idx, row],
                reverse=self.order_by[1] == SqlOrderBy.DESC,
            )
            for idx, row in self.seed_records[table].iterrows()
        ]
        heapq.heapify(queue)

        conditions = self.conditions.get(table)

        query = QueryWhereTransformer(self.tables[table]).transform(
            conditions
        ) if conditions else None

        if reset_matches:
            for table in self.tables:
                self.matches[table] = UnorderedTupleSet()
                self.not_matches[table] = UnorderedTupleSet()

        comparisons = 0
        emitted = 0

        while len(queue):
            item = heapq.heappop(queue)
            if item.solved:
                if not conditions or len(pd.DataFrame([item.item[0]]).query(query)):
                    self.emit(item.item[0], item.item[1], comparisons)
                    emitted += 1
                    if self.top_k and emitted >= self.top_k:
                        break
                continue
            if item.item[0] in analyzed:
                continue

            entity_cluster, comparisons = self.match(
                table, analyzed, item.item[0], comparisons
            )

            entity, ids = self.resolve_entity(table, entity_cluster)
            if len(entity):
                heapq.heappush(
                    queue,
                    PriorityItem(
                        entity[self.order_by[0].name],
                        [entity, ids],
                        self.order_by[1] == SqlOrderBy.DESC,
                        solved=True,
                    ),
                )

    def match(self, table: Table, analyzed: set, record_idx: Any, comparisons: int):
        to_analyze = deque([record_idx])
        entity_cluster = set([record_idx])

        while to_analyze:
            idx = to_analyze.popleft()

            if idx  in analyzed:
                continue
            
            candidates = [
                block.index for block in table.blocks if record_idx in block.index
            ]
            if candidates:
                block = set().union(*candidates)
            else:
                block = set((record_idx,))
            
            for candidate in block:
                if candidate in analyzed:
                    continue
                if candidate in entity_cluster:
                    continue
                result = self.is_match(table, idx, candidate, comparisons)
                comparisons = result[1]
                if result[0]:
                    entity_cluster.add(candidate)
                    to_analyze.append(candidate)

        analyzed.update(entity_cluster)
        return entity_cluster, comparisons

    def is_match(self, table: Table, l, r, comparisons):
        if l == r:
            return True, comparisons
        if (l, r) in self.matches[table]:
            return True, comparisons
        if (l, r) in self.not_matches[table]:
            return False, comparisons
        comparisons += 1
        match = table.matcher(table.data.loc[l], table.data.loc[r])
        if match:
            self.matches[table].add((l, r))
        else:
            self.not_matches[table].add((l, r))
        return match, comparisons

    def resolve_entity(self, table: Table, entity_cluster: set):
        entity = {}
        records = table.data.loc[table.data.index.isin(entity_cluster)]
        for column in self.tables[table]:
            entity[column.name] = column.resolution_function.resolve(
                records[column.name]
            )
        return pd.Series(entity), set(records[table.id_column])
