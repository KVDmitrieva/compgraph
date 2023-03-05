from __future__ import annotations

import typing as tp

from . import operations as ops
from .external_sort import ExternalSort


class EmptyOperation(Exception):
    pass


class Graph:
    """Computational graph implementation"""
    def __init__(self, *args: tp.Any) -> None:
        self._operation: ops.Operation | None = None
        self._graphs: tp.Any = args

    @staticmethod
    def graph_from_iter(name: str) -> Graph:
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph._operation = ops.ReadIterFactory(name)
        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> Graph:
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph._operation = ops.Read(filename, parser)
        return graph

    def map(self, mapper: ops.Mapper) -> Graph:
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        graph = Graph(self)
        graph._operation = ops.Map(mapper)
        return graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        graph = Graph(self)
        graph._operation = ops.Reduce(reducer, keys)
        return graph

    def sort(self, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        graph = Graph(self)
        graph._operation = ExternalSort(keys)
        return graph

    def join(self, joiner: ops.Joiner, join_graph: Graph, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = Graph(self, join_graph)
        graph._operation = ops.Join(joiner, keys)
        return graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        if self._operation is None:
            raise EmptyOperation('No operation passed!')

        if not len(self._graphs):
            yield from self._operation(**kwargs)
        elif len(self._graphs) == 1:
            yield from self._operation(self._graphs[0].run(**kwargs))
        else:
            yield from self._operation(self._graphs[0].run(**kwargs),
                                       self._graphs[1].run(**kwargs))
