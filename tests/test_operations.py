import copy
import dataclasses
import json
import typing as tp
from pathlib import Path

import pytest

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.BinaryArithmeticOperation(operation=lambda row: row['first'] * row['second'], column='result'),
        data=[
            {'test_id': 1, 'first': 10, 'second': 5},
            {'test_id': 2, 'first': 0, 'second': 8}
        ],
        ground_truth=[
            {'test_id': 1, 'first': 10, 'second': 5, 'result': 50},
            {'test_id': 2, 'first': 0, 'second': 8, 'result': 0}
        ],
        cmp_keys=('test_id', 'first', 'second', 'result')
    ),
    MapCase(
        mapper=ops.Haversine(start='start', end='end', column='result'),
        data=[
            {'test_id': 1, 'start': [37.84870228730142, 55.73853974696249],
             'end': [37.8490418381989, 55.73832445777953]},
            {'test_id': 2, 'start': [37.524768467992544, 55.88785375468433],
             'end': [37.52415172755718, 55.88807155843824]}
        ],
        ground_truth=[
            {'test_id': 1, 'start': [37.84870228730142, 55.73853974696249],
             'end': [37.8490418381989, 55.73832445777953], 'result': pytest.approx(0.032, rel=1e-3)},
            {'test_id': 2, 'start': [37.524768467992544, 55.88785375468433],
             'end': [37.52415172755718, 55.88807155843824], 'result': pytest.approx(0.0455, rel=1e-3)}
        ],
        cmp_keys=('test_id', 'start', 'end', 'result')
    ),
    MapCase(
        mapper=ops.DatetimeExtractor(date_column='enter_time', date_formate='%Y', column='result'),
        data=[
            {'test_id': 1, 'enter_time': '20171020T112237.427000'},
            {'test_id': 2, 'enter_time': '20181011T145551'},
        ],
        ground_truth=[
            {'test_id': 1, 'enter_time': '20171020T112237.427000', 'result': '2017'},
            {'test_id': 2, 'enter_time': '20181011T145551', 'result': '2018'},
        ],
        cmp_keys=('test_id', 'enter_time', 'result')
    ),
    MapCase(
        mapper=ops.Duration(start='start', end='end', column='result'),
        data=[
            {'test_id': 1, 'start': '20171020T112237.427000', 'end': '20171020T112238.723000'},
            {'test_id': 2, 'start': '20171011T145551.957000', 'end': '20171011T145553.040000'},
        ],
        ground_truth=[
            {'test_id': 1, 'start': '20171020T112237.427000',
             'end': '20171020T112238.723000', 'result': pytest.approx(0.00036)},
            {'test_id': 2, 'start': '20171011T145551.957000',
             'end': '20171011T145553.040000', 'result':  pytest.approx(0.00030083, rel=1e-4)},
        ],
        cmp_keys=('test_id', 'start', 'end', 'result')
    ),
    MapCase(
        mapper=ops.StrToInt(['num']),
        data=[
            {'test_id': 1, 'num': '15'},
            {'test_id': 2, 'num': '01'}
        ],
        ground_truth=[
            {'test_id': 1, 'num': 15},
            {'test_id': 2, 'num': 1}
        ],
        cmp_keys=('test_id', 'num')
    ),
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_ground_truth_rows, key=key_func) == sorted(mapper_result, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.ground_truth, key=key_func) == sorted(result, key=key_func)


tests = [
    {'test_id': 1, 'text': 'hello'},
    {'test_id': 2, 'text': 'world'}
]


def test_read_iter() -> None:
    result = ops.ReadIterFactory('test')(test=lambda: iter(tests))
    assert list(result) == tests


@pytest.fixture(scope='session')
def data_file(tmp_path_factory) -> Path:  # type: ignore
    path = tmp_path_factory.mktemp('data') / 'tmp.txt'
    with open(path, 'w') as fn:
        for test in tests:
            print(json.dumps(test), file=fn)
    return path


def test_read(data_file: Path) -> None:
    result = ops.Read(data_file.as_posix(), ops.json_parser)()
    assert list(result) == tests
