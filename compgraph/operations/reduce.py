import heapq
import typing as tp
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import deepcopy
from itertools import groupby

from .base import Operation
from .base import TRowsIterable, TRowsGenerator


__all__ = ['Reducer', 'Reduce', 'FirstReducer', 'TopN', 'TermFrequency', 'Count', 'Sum']


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for _, group in groupby(rows, key=lambda row: [row[k] for k in self.keys]):
            yield from self.reducer(tuple(self.keys), group)


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in heapq.nlargest(self.n, rows, key=lambda r: r[self.column_max]):
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words: tp.DefaultDict[str, int] = defaultdict(int)
        total = 0
        row = {}
        for row in rows:
            total += 1
            words[row[self.words_column]] += 1

        new_row = {}
        for key in group_key:
            new_row[key] = row[key]

        for word, val in words.items():
            new_row[self.words_column] = word
            new_row[self.result_column] = val / total
            yield deepcopy(new_row)


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        row = {}
        new_row = {self.column: 0}
        for row in rows:
            new_row[self.column] += 1
        for key in group_key:
            new_row[key] = row[key]
        yield new_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        row = {}
        new_row = {self.column: 0}
        for row in rows:
            new_row[self.column] += row[self.column]

        for key in group_key:
            new_row[key] = row[key]

        yield new_row
