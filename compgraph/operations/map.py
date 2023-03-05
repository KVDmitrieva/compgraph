import datetime
import math
import re
import typing as tp
from abc import abstractmethod, ABC
from copy import deepcopy
from string import punctuation

from .base import Operation
from .base import TRow, TRowsIterable, TRowsGenerator


__all__ = ['Mapper', 'Map', 'DummyMapper', 'FilterPunctuation',
           'LowerCase', 'Split', 'Product', 'Filter', 'Project',
           'Haversine', 'DatetimeExtractor', 'StrToInt',
           'BinaryArithmeticOperation', 'Duration']


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(deepcopy(row))


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = ''.join((filter(lambda s: s not in punctuation, row[self.column])))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = LowerCase._lower_case(row[self.column])  # row[self.column].lower()
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str = r'\s+') -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        s = row.pop(self.column)
        while re.search(self.separator, s) is not None:
            val, s = re.split(self.separator, s, 1)
            row[self.column] = val
            yield deepcopy(row)

        row[self.column] = s
        yield row


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = 1
        for col in self.columns:
            row[self.result_column] *= row[col]
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        new_row = {}
        for col in self.columns:
            new_row[col] = row[col]
        yield new_row


class BinaryArithmeticOperation(Mapper):
    """Calculate result of operation between two columns"""
    def __init__(self, operation: tp.Callable[[TRow], float], column: str) -> None:
        """
        :param operation: binary operation
        :param column: name of result column
        """
        self.operation = operation
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self.operation(row)
        yield row


class Haversine(Mapper):
    """Calculate haversine distance"""
    def __init__(self, start: str, end: str, column: str) -> None:
        """
        :param start: name of column with starting coordinates
        :param end: name of column with ending coordinates
        :param column: name of result column
        """
        self.start = start
        self.end = end
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        lon1, lat1 = map(math.radians, row[self.start])
        lon2, lat2 = map(math.radians, row[self.end])

        lat_sin = math.sin((lat2 - lat1) / 2) ** 2
        long_sin = math.sin((lon2 - lon1) / 2) ** 2

        angle = math.sqrt(lat_sin + math.cos(lat1) * math.cos(lat2) * long_sin)
        earth_radius = 6373
        row[self.column] = 2 * earth_radius * math.asin(angle)

        yield row


class DatetimeExtractor(Mapper):
    """Extract information from datetime"""
    def __init__(self, date_column: str, date_formate: str,  column: str) -> None:
        """
        :param date_column: name of datetime column
        :param date_formate: format of datetime to extract info
        :param column: name of result column
        """
        self.date_column = date_column
        self.date_format = date_formate
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        str_format = '%Y%m%dT%H%M%S.%f' if '.' in row[self.date_column] else '%Y%m%dT%H%M%S'
        date = datetime.datetime.strptime(row[self.date_column], str_format)
        row[self.column] = date.strftime(self.date_format)
        yield row


class Duration(Mapper):
    """Calculate duration between two datetime columns in hours"""
    def __init__(self, start: str, end: str, column: str) -> None:
        """
        :param start: name of start datetime column
        :param end: name of end datetime column
        :param column: name of result column
        """
        self.start = start
        self.end = end
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        str_format = '%Y%m%dT%H%M%S.%f' if '.' in row[self.start] else '%Y%m%dT%H%M%S'
        d1 = datetime.datetime.strptime(row[self.start], str_format)
        str_format = '%Y%m%dT%H%M%S.%f' if '.' in row[self.end] else '%Y%m%dT%H%M%S'
        d2 = datetime.datetime.strptime(row[self.end], str_format)
        row[self.column] = (d2 - d1).total_seconds() / 3600
        yield row


class StrToInt(Mapper):
    """Converts columns' value from str to int"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: name of columns to be converted
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        for col in self.columns:
            row[col] = int(row[col])
        yield row
