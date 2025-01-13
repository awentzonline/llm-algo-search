import pickle

import click
import numpy as np


@click.command()
@click.argument('proposal_filename')
def main(proposal_filename):
    with open(proposal_filename, 'rb') as infile:
        history = pickle.load(infile)

    for prop in history:
        print(prop.raw)
        print(prop.eval_results)


if __name__ == '__main__':
    main()
