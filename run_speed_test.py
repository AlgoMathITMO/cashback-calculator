from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from myutils.json import save_json

from config import TESTS_DIR
from cashback_calculator.db import Card, N_MCCS, ALL_MCCS, CARD_IDS
from cashback_calculator.two_card_strategy_optimisation import optimize_cashback_for_cards

RESULTS_JSON_FPATH = TESTS_DIR / 'speed_test_results.json'
RESULTS_FIGURE_FPATH = TESTS_DIR / 'speed_test_results.pdf'


def generate_spendings_vector(n_values: int, total: float = 120000) -> np.ndarray:
    vector = np.random.uniform(size=n_values)
    vector = vector / vector.sum() * total
    
    return vector


def main():
    results = []

    for _ in trange(10):
        for n in range(2, N_MCCS + 1, 20):
            mccs = np.random.choice(ALL_MCCS, size=n, replace=False).tolist()
            vector = generate_spendings_vector(n)

            cards = [Card(card_id, mccs) for card_id in CARD_IDS]

            for c1, c2 in combinations(cards, 2):
                t1 = datetime.now()
                optimize_cashback_for_cards(vector, c1, c2)
                t2 = datetime.now()
                delta = (t2 - t1).total_seconds()

                results.append((n, delta))

    save_json(results, RESULTS_JSON_FPATH)
    
    results = pd.DataFrame(results, columns=['n', 't'])

    sns.lineplot(data=results, x='n', y='t')

    plt.xlabel('number of MCCs')
    plt.ylabel('run time (seconds)')
    plt.title('Run time of cashback optimisation')

    plt.grid(ls='dotted')

    plt.savefig(RESULTS_FIGURE_FPATH, bbox_inches='tight')


if __name__ == '__main__':
    main()
