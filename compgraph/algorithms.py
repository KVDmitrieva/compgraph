import typing as tp
from math import log

from . import Graph, operations


def get_head(input_stream_name: str,
             parser: tp.Callable[[str], operations.TRow] | None = None) -> Graph:
    if parser is None:
        input_graph = Graph.graph_from_iter(input_stream_name)
    else:
        input_graph = Graph.graph_from_file(input_stream_name, parser)
    return input_graph


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count',
                     parser: tp.Callable[[str], operations.TRow] | None = None) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    input_graph = get_head(input_stream_name, parser)

    return input_graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id',
                         text_column: str = 'text', result_column: str = 'tf_idf',
                         parser: tp.Callable[[str], operations.TRow] | None = None) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    input_graph = get_head(input_stream_name, parser)

    doc_count_column = 'doc_count'
    total_column = 'total'

    split_graph = input_graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    doc_graph = input_graph \
        .sort([doc_column]) \
        .reduce(operations.FirstReducer(), [doc_column]) \
        .reduce(operations.Count(total_column), [])

    idf_column = 'idf'
    idf_graph = split_graph \
        .sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count(doc_count_column), [text_column]) \
        .join(operations.InnerJoiner(), doc_graph, []) \
        .map(operations.BinaryArithmeticOperation(lambda row: log(row[total_column] / row[doc_count_column]),
                                                  idf_column))

    tf_graph = split_graph \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column])

    res_graph = tf_graph \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), idf_graph, [text_column]) \
        .map(operations.Product([idf_column, 'tf'], result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column])

    return res_graph


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id',
              text_column: str = 'text', result_column: str = 'pmi',
              parser: tp.Callable[[str], operations.TRow] | None = None) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    input_graph = get_head(input_stream_name, parser)

    split_graph = input_graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .map(operations.Filter(lambda row: len(row[text_column]) > 4))

    doc_tf_column = 'doc_tf'
    freq_graph = split_graph.sort([doc_column, text_column]) \
        .reduce(operations.Count(doc_tf_column), [doc_column, text_column]) \
        .map(operations.Filter(lambda row: row[doc_tf_column] > 1)) \

    filtered_graph = split_graph.sort([doc_column, text_column]) \
        .join(operations.InnerJoiner(), freq_graph, [doc_column, text_column]) \

    doc_tf_graph = filtered_graph \
        .reduce(operations.TermFrequency(text_column, doc_tf_column), [doc_column])

    tf_column = 'total_tf'
    total_tf_graph = filtered_graph \
        .reduce(operations.TermFrequency(text_column, tf_column), []) \
        .sort([text_column])

    calc_pmi_graph = doc_tf_graph.sort([text_column]) \
        .join(operations.InnerJoiner(), total_tf_graph, [text_column]) \
        .map(operations.BinaryArithmeticOperation(lambda row: log(row[doc_tf_column] / row[tf_column]),
                                                  result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column])

    return calc_pmi_graph


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed',
                      parser: tp.Callable[[str], operations.TRow] | None = None) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    input_time_graph = get_head(input_stream_name_time, parser)
    input_len_graph = get_head(input_stream_name_length, parser)

    haversine_column = 'haversine'
    hav_graph = input_len_graph \
        .map(operations.Haversine(start_coord_column, end_coord_column, haversine_column)) \
        .map(operations.Project([edge_id_column, haversine_column])) \
        .sort([edge_id_column])

    duration_column = 'duration'
    time_graph = input_time_graph \
        .map(operations.DatetimeExtractor(enter_time_column, '%a', weekday_result_column)) \
        .map(operations.DatetimeExtractor(enter_time_column, '%H', hour_result_column)) \
        .map(operations.StrToInt([hour_result_column])) \
        .map(operations.Duration(enter_time_column, leave_time_column, duration_column)) \
        .map(operations.Project([edge_id_column, weekday_result_column, hour_result_column, duration_column])) \
        .sort([edge_id_column])

    joint_graph = time_graph \
        .join(operations.InnerJoiner(), hav_graph, [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column])

    duration_graph = joint_graph \
        .reduce(operations.Sum(duration_column), [edge_id_column, weekday_result_column, hour_result_column])

    distance_graph = joint_graph \
        .reduce(operations.Sum(haversine_column), [edge_id_column, weekday_result_column, hour_result_column])

    speed_graph = duration_graph\
        .join(operations.InnerJoiner(), distance_graph, [edge_id_column, weekday_result_column, hour_result_column]) \
        .map(operations.BinaryArithmeticOperation(lambda row: row[haversine_column] / row[duration_column],
                                                  speed_result_column)) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))
    return speed_graph
