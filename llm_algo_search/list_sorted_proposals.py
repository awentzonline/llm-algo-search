import pickle

import click
import numpy as np


@click.command()
@click.argument('proposal_filename')
def main(proposal_filename):
    with open(proposal_filename, 'rb') as infile:
        history = pickle.load(infile)

    scores = []
    for i, prop in enumerate(history):
        if prop.eval_results is None:
            worst_accuracy = -1
        else:
            worst_accuracy = np.min(prop.eval_results['accuracy'])
        scores.append(worst_accuracy)

    sorted_ids = reversed(np.argsort(scores))
    for id in sorted_ids:
        prop = history[id]
        print(prop.format_proposal())
        print(f'Proposal #{id}')
        input('>> Press enter for next best <<')


if __name__ == '__main__':
    main()
