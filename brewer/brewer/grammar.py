from lark import Lark, Transformer, Token
from .column import Column

query_where_grammar = """
start: _sep{boolean_term, "or"i} -> _any
boolean_term: _sep{boolean_factor, "and"i} -> _all
boolean_factor: "not"i boolean_factor -> _not
              | column comp_op expression -> _comp
              | "(" start ")" -> _par
column: table_alias "." column_name
column_name: CNAME
table_alias: CNAME
expression: value
          | bool
bool: "true"i  -> _true
    | "false"i -> _false
value: ESCAPED_STRING | SIGNED_NUMBER
comp_op: "=" -> _eq
       | ">" -> _gt
       | "<" -> _lt
       | ">=" -> _gte
       | "<="  -> _lte
       | "<>" -> _neq
       | "like"i -> _like
       | "ilike"i -> _ilike

_sep{x, sep}: x (sep x)*

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.CNAME
%import common.WS
%ignore WS
"""


class QueryWhereTransformer(Transformer):
    def __init__(
        self,
        columns: list[Column] = [],
        extract_seed_records: bool = False,
    ):
        super().__init__()
        self.columns = columns
        self.extract_seed_records = extract_seed_records

    def _any(self, items):
        items = [item for item in items if item is not None]
        if len(items) > 1:
            return f"({' or '.join(items)})"
        if len(items):
            return items[0]

    def _all(self, items):
        items = [item for item in items if item is not None]
        if len(items) > 1:
            if self.extract_seed_records:
                return f"({' or '.join(items)})"
            return f"({' and '.join(items)})"
        if len(items):
            return items[0]

    def _par(self, items):
        return items[0]

    def _not(self, items):
        if items[0] is not None:
            return f"(not {items[0]})"

    def _comp(self, items):
        if items[0] is not None:
            if self.extract_seed_records:
                if items[1] == " == ":
                    return items[0][1].extract_seed_records_eq(items[0][0], items[2])
                if items[1] == " != ":
                    return items[0][1].extract_seed_records_neq(items[0][0], items[2])
                if items[1] == " > ":
                    return items[0][1].extract_seed_records_gt(items[0][0], items[2])
                if items[1] == " >= ":
                    return items[0][1].extract_seed_records_gte(items[0][0], items[2])
                if items[1] == " < ":
                    return items[0][1].extract_seed_records_lt(items[0][0], items[2])
                if items[1] == " <= ":
                    return items[0][1].extract_seed_records_lte(items[0][0], items[2])
                if items[1][-1] == "=":
                    return items[0][1].extract_seed_records_like(items[0][0], items[2])
                if items[1][-1] == "e":
                    return items[0][1].extract_seed_records_ilike(items[0][0], items[2])
            if items[1][0] == ".":
                items[2] += ")"
            return "".join(items)

    def column(self, items):
        column_name = f"{items[0].children[0]}_{items[1].children[0]}"

        if column_name in [column.name for column in self.columns]:
            if self.extract_seed_records:
                resolution_function = [
                    column.resolution_function
                    for column in self.columns
                    if column.name == column_name
                ]

                return (
                    column_name,
                    resolution_function[0],
                )
            return column_name

    def _true(self, _):
        return "true"

    def _false(self, _):
        return "false"

    def value(self, items: list[Token]):
        token = items[0]
        if token.type == "SIGNED_NUMBER":
            return token.value
        return token.value.replace("%", ".*").replace("_", ".")

    def _eq(self, _):
        return " == "

    def _gt(self, _):
        return " > "

    def _lt(self, _):
        return " < "

    def _gte(self, _):
        return " >= "

    def _lte(self, _):
        return " <= "

    def _neq(self, _):
        return " != "

    def _like(self, _):
        return ".str.contains(na=False, pat="

    def _ilike(self, _):
        return ".str.contains(na=False, case=False, pat="

    def expression(self, items):
        return items[0]


query_where_parser = Lark(query_where_grammar)

query_on_grammar = """
start: column "=" column -> _on
column: table_alias "." column_name
column_name: CNAME
table_alias: CNAME
%import common.CNAME
%import common.WS
%ignore WS
"""


class QueryOnTransformer(Transformer):
    def _on(self, items):
        return {items[0]: items[1], items[1]: items[0]}

    def column(self, items):
        return f"{items[0].children[0]}_{items[1].children[0]}"


query_on_parser = Lark(query_on_grammar)
