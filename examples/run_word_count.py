import json

import click

from compgraph.algorithms import word_count_graph
from compgraph.operations import json_parser


@click.command()
@click.argument('input_filepath', type=str)
@click.argument('output_filepath', type=str)
def main(input_filepath: str, output_filepath: str) -> None:
    graph = word_count_graph(input_stream_name=input_filepath, parser=json_parser)

    result = graph.run()
    with open(output_filepath, 'w') as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == '__main__':
    main()
