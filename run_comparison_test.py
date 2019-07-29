from itertools import combinations

import pandas as pd
from matplotlib import pyplot as plt

from config import DATA_DIR, TESTS_DIR
from cashback_calculator.db import Card, CARD_IDS, ALL_MCCS
from cashback_calculator.two_card_strategy_optimisation import optimize_cashback_for_cards

DATA_FPATH = DATA_DIR / 'money_movement.csv'
RESULTS_CSV_FPATH = TESTS_DIR / 'comparison_test_results.csv'
RESULTS_FIGURE_FPATH = TESTS_DIR / 'comparison_test_results.pdf'


def main():
    data = pd.read_csv(DATA_FPATH, parse_dates=['date'], dtype={'mcc': str})
    data['month'] = data['date'].apply(lambda dt: dt.replace(day=1))

    months = data['month'].unique()

    real_cashback = data[data['is_cashback']].groupby('month')['income'].sum().reindex(months)

    monthly_expense_vectors = data[~data['is_cashback']] \
        .groupby(['month', 'mcc'])['expense'].sum().reset_index() \
        .pivot(columns='mcc', index='month', values='expense').reindex(columns=ALL_MCCS).fillna(0) \
        .reindex(months)

    cashback = pd.DataFrame({'real cashback': real_cashback})

    for month, next_month in zip(months, months[1:]):
        vector = monthly_expense_vectors.loc[month]

        mccs = vector[vector > 0].index.tolist()
        vector = vector[mccs]

        cards = [Card(card_id, all_mccs=mccs) for card_id in CARD_IDS]

        for card in cards:
            key = card.card_name
            profit = card.calculate_profit(vector)
            cashback.loc[next_month, key] = profit

        for card1, card2 in combinations(cards, 2):
            key = f'{card1.card_name} + {card2.card_name}'
            profit = optimize_cashback_for_cards(vector, card1, card2)[-1]
            cashback.loc[next_month, key] = profit

    cashback = cashback.T
    idx = cashback.sum(axis=1).sort_values(ascending=False).index
    cashback = cashback.loc[idx]

    cashback.to_csv(RESULTS_CSV_FPATH)

    cashback.loc['real cashback'].plot(ls='dashed')

    for _, row in cashback.iloc[:5].iterrows():
        row.plot()

    plt.xlabel('month')
    plt.ylabel('cashback profit')
    plt.title('Real vs. optimal cashback profit')

    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))

    plt.savefig(RESULTS_FIGURE_FPATH, bbox_inches='tight')


if __name__ == '__main__':
    main()
