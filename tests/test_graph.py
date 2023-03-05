import json
from pathlib import Path

import pytest

from compgraph import Graph
from compgraph import operations as ops


test_data = [
    {'test_id': 1, 'text': 'hello'},
    {'test_id': 2, 'text': 'world'}
]


@pytest.fixture(scope='session')
def data_file(tmp_path_factory) -> Path:  # type: ignore
    path = tmp_path_factory.mktemp('data') / 'tmp.txt'
    with open(path, 'w') as fn:
        for test in test_data:
            print(json.dumps(test), file=fn)
    return path


def test_graph_from_file(data_file: Path) -> None:
    graph = Graph.graph_from_file(data_file.as_posix(), ops.json_parser)
    result = graph.run()

    assert list(result) == test_data


def test_graph_from_iter() -> None:
    tests = [
        {'test_id': 1, 'text': 'Hello, world!'},
        {'test_id': 2, 'text': 'Hello darkness, my old friend'}
    ]

    expected = [
        {'test_id': 1, 'text': 'Hello, world!'},
        {'test_id': 2, 'text': 'Hello darkness, my old friend'}
    ]

    graph = Graph.graph_from_iter('test')
    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_map() -> None:
    tests = [
        {'test_id': 1, 'text': 'Hello, world!'},
        {'test_id': 2, 'text': 'Hello darkness, my old friend'}
    ]

    expected = [
        {'test_id': 1, 'text': 'Hello,'},
        {'test_id': 1, 'text': 'world!'},
        {'test_id': 2, 'text': 'Hello'},
        {'test_id': 2, 'text': 'darkness,'},
        {'test_id': 2, 'text': 'my'},
        {'test_id': 2, 'text': 'old'},
        {'test_id': 2, 'text': 'friend'}
    ]
    graph = Graph.graph_from_iter('test').map(ops.Split(column='text'))
    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_reduce() -> None:
    tests = [
        {'test_id': 1, 'text': 'First!'},
        {'test_id': 1, 'text': 'Second!'},

        {'test_id': 2, 'text': 'Hello darkness, my old friend'}
    ]

    expected = [
        {'test_id': 1, 'text': 'First!'},

        {'test_id': 2, 'text': 'Hello darkness, my old friend'}
    ]
    graph = Graph.graph_from_iter('test').reduce(ops.FirstReducer(), keys=('test_id', ))
    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_sort() -> None:
    tests = [
        {'test_id': 1, 'text': 'banana'},
        {'test_id': 2, 'text': 'orange'},
        {'test_id': 3, 'text': 'apple'}
    ]

    expected = [
        {'test_id': 3, 'text': 'apple'},
        {'test_id': 1, 'text': 'banana'},
        {'test_id': 2, 'text': 'orange'}
    ]
    graph = Graph.graph_from_iter('test').sort(['text'])
    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_join() -> None:
    data1 = [
        {'test_id': 1, 'fruit': 'banana'},
        {'test_id': 2, 'fruit': 'orange'},
        {'test_id': 3, 'fruit': 'apple'}
    ]

    data2 = [
        {'price': 11, 'fruit': 'banana'},
        {'price': 24, 'fruit': 'orange'},
        {'price': 35, 'fruit': 'apple'}
    ]

    expected = [
        {'test_id': 1, 'fruit': 'banana', 'price': 11},
        {'test_id': 2, 'fruit': 'orange', 'price': 24},
        {'test_id': 3, 'fruit': 'apple', 'price': 35}
    ]

    join_graph = Graph.graph_from_iter('data_right')

    graph = Graph.graph_from_iter('data_left') \
        .join(ops.InnerJoiner(), join_graph, ['fruit'])

    result = graph.run(data_left=lambda: iter(data1),
                       data_right=lambda: iter(data2))

    assert list(result) == expected
