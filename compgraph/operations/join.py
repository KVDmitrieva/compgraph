import typing as tp
from abc import abstractmethod, ABC
from copy import deepcopy
from itertools import groupby

from .base import Operation
from .base import TRowsIterable, TRowsGenerator


__all__ = ['Joiner', 'Join', 'InnerJoiner', 'OuterJoiner', 'LeftJoiner', 'RightJoiner']


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    @staticmethod
    def _get_next_group(groups: tp.Iterator[tp.Any]) -> tp.Any:
        try:
            next_group = next(groups)
            return next_group
        except StopIteration:
            return None

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        left_groups = groupby(rows, key=lambda row: [row[k] for k in self.keys])
        right_groups = groupby(args[0], key=lambda row: [row[k] for k in self.keys])

        left_next = self._get_next_group(left_groups)
        right_next = self._get_next_group(right_groups)

        while (left_next is not None) and (right_next is not None):
            if left_next[0] == right_next[0]:
                yield from self.joiner(self.keys, left_next[1], right_next[1])
                left_next = self._get_next_group(left_groups)
                right_next = self._get_next_group(right_groups)
            elif left_next[0] < right_next[0]:
                yield from self.joiner(self.keys, left_next[1], [])
                left_next = self._get_next_group(left_groups)
            else:
                yield from self.joiner(self.keys, [], right_next[1])
                right_next = self._get_next_group(right_groups)

        while left_next is not None:
            yield from self.joiner(self.keys, left_next[1], [])
            left_next = self._get_next_group(left_groups)

        while right_next is not None:
            yield from self.joiner(self.keys, [], right_next[1])
            right_next = self._get_next_group(right_groups)


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        all_rows_b = [row for row in rows_b]

        for row_a in rows_a:
            for row_b in all_rows_b:
                new_row = deepcopy(row_b)

                same_cols = set(new_row.keys()) & set(row_a.keys()) - set(keys)
                for col in same_cols:
                    row_a[col + self._a_suffix] = row_a.pop(col)
                    new_row[col + self._b_suffix] = new_row.pop(col)

                new_row.update(row_a)
                yield new_row


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        all_rows_b = [row for row in rows_b]
        if not len(all_rows_b):
            yield from rows_a
        else:
            a_empty = True
            for row_a in rows_a:
                a_empty = False
                for row_b in all_rows_b:
                    new_row = deepcopy(row_b)

                    same_cols = set(new_row.keys()) & set(row_a.keys()) - set(keys)
                    for col in same_cols:
                        row_a[col + self._a_suffix] = row_a.pop(col)
                        new_row[col + self._b_suffix] = new_row.pop(col)

                    new_row.update(row_a)
                    yield new_row

            if a_empty:
                yield from all_rows_b


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        all_rows_b = [row for row in rows_b]
        if not len(all_rows_b):
            yield from rows_a
        else:
            for row_a in rows_a:
                for row_b in all_rows_b:
                    new_row = deepcopy(row_b)

                    same_cols = set(new_row.keys()) & set(row_a.keys()) - set(keys)
                    for col in same_cols:
                        row_a[col + self._a_suffix] = row_a.pop(col)
                        new_row[col + self._b_suffix] = new_row.pop(col)

                    new_row.update(row_a)
                    yield new_row


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        all_rows_b = [row for row in rows_b]
        a_empty = True
        for row_a in rows_a:
            a_empty = False
            for row_b in all_rows_b:
                new_row = deepcopy(row_b)

                same_cols = set(new_row.keys()) & set(row_a.keys()) - set(keys)
                for col in same_cols:
                    row_a[col + self._a_suffix] = row_a.pop(col)
                    new_row[col + self._b_suffix] = new_row.pop(col)

                new_row.update(row_a)
                yield new_row

        if a_empty:
            yield from all_rows_b
