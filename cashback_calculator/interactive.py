import warnings
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ipywidgets
from IPython.display import display, HTML

from cashback_calculator.db import Card, N_MCCS, ALL_MCCS, MCC2ID, MCC_CATEGORIES, CARD_IDS
from cashback_calculator.two_card_strategy_optimisation import optimize_cashback

__all__ = [
    'Interact',
]

warnings.filterwarnings('ignore')

CATEGORY_NAMES = list(MCC_CATEGORIES.keys())

MCC2CATEGORY = {
    mcc: category
    for category, mccs in MCC_CATEGORIES.items()
    for mcc in mccs
}

CARDS = [Card(card_id) for card_id in CARD_IDS]
CARD_NAMES = ['None'] + [card.card_name for card in CARDS]


def get_vector(amount_spent_by_category) -> np.ndarray:
    vector = np.zeros(N_MCCS)
    
    for key, value in amount_spent_by_category.items():
        if value > 0:
            value -= 0.01
            mccs = MCC_CATEGORIES[key]
            n_mccs = len(mccs)
            mask = list(map(MCC2ID.get, mccs))
            vector[mask] = value / n_mccs
            
    return vector


def get_categories(vector: np.ndarray) -> pd.Series:
    vector = pd.Series(vector, index=ALL_MCCS)
    vector.index = vector.index.map(MCC2CATEGORY)
    
    amount_spent_by_category = vector.groupby(level=0).sum().reindex(CATEGORY_NAMES).fillna(0)
    amount_spent_by_category = amount_spent_by_category[amount_spent_by_category > 0]
    
    return amount_spent_by_category


def plot_cashback_cases(profit: float, vector: np.ndarray, c1: Card, c2: Card, my_card: Optional[Card] = None):
    cashback_cases = pd.Series({
        'optimal': profit,
        'only first card': c1.calculate_profit(vector),
        'only second card': c2.calculate_profit(vector),
    })
    
    if my_card:
        cashback_cases['your card'] = my_card.calculate_profit(vector)
        
    plt.figure(figsize=(6, 2.5))
        
    plt.bar(cashback_cases.index, cashback_cases.values, zorder=10, color=[f'C{i}' for i in range(cashback_cases.size)])
    
    ymax = cashback_cases.max()
    
    for i, value in enumerate(cashback_cases.values):
        plt.text(i, ymax / 50, str(round(value)), ha='center', va='bottom', zorder=11)
    
    plt.grid(axis='y', ls='dotted')
    plt.xticks(rotation=10)
    plt.title('Optimal cashback')
    
    plt.show()
    plt.close()
    
    
def plot_strategy(vector: np.ndarray, x: np.ndarray, c1: Card, c2: Card):
    fig = plt.figure(figsize=(8, 3.5))
    gs = fig.add_gridspec(ncols=3, nrows=1)    
    fig.subplots_adjust(top=0.92, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, -1])
    
    c1_categories = get_categories(vector * x)
    c2_categories = get_categories(vector * (1 - x))
    
    xmax = max(c1_categories.max(), c2_categories.max())
    xmax *= 6 / 5
    ax1.set_xlim(-xmax, xmax)

    ax1.barh(c1_categories.index, -c1_categories.values, zorder=8, label=c1.card_name)
    ax1.barh(c2_categories.index, c2_categories.values, zorder=8, label=c2.card_name)
    
    for i, value in enumerate(c1_categories.values):
        ax1.text(- xmax / 50, i, str(round(value)), ha='right', va='center', fontsize=8, zorder=9)
        
    for i, value in enumerate(c2_categories.values):
        ax1.text(xmax / 50, i, str(round(value)), ha='left', va='center', fontsize=8, zorder=9)
    
    ax1.legend(prop={'size': 8}, loc='upper center', bbox_to_anchor=(0.8, -0.15)).set_zorder(15)
    ax1.grid(ls='dotted', axis='x')
    
    ax1.axvline(0, c='black', lw=0.8).set_zorder(10)
    
    xticks = ax1.get_xticks()[1:-1]
    xticklabels = np.abs(xticks).astype(int)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    
    ax1.set_xlabel('rubles')

    #ax1.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    
    # ax1.set_ylim(bottom=ax1.get_ylim()[0] - 0.5)
    ax1.invert_yaxis()
    
    full_categories = get_categories(vector)
    c1_categories_frac = c1_categories / full_categories
    c2_categories_frac = 1 - c1_categories_frac
    
    ax2.barh(c1_categories_frac.index, c1_categories_frac.values)
    ax2.barh(c2_categories_frac.index, c2_categories_frac.values, left=c1_categories_frac.values)
    
    ax2.set_xlim(0, 1)
    
    for i, value in enumerate(c1_categories_frac):
        ax2.text(0.02, i, str(round(value * 100)) + '%', ha='left', va='center', fontsize=8, zorder=9)
        ax2.text(0.98, i, str(round((1 - value) * 100)) + '%', ha='right', va='center', fontsize=8, zorder=9)
    
    xticks = np.arange(0, 1.01, 0.2).round(1)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks)
    
    ax2.set_xlabel('%')
    
    ax2.tick_params(left=False, labelleft=False, right=True, labelright=True)
    
    # ax2.set_ylim(bottom=ax2.get_ylim()[0] - 0.5)
    ax2.invert_yaxis()
    
    fig.suptitle('Optimal spending strategy')
    
    plt.show()
    plt.close()




class Interact:
    DEFAULT_VALUES = {'restaurants': 15015, 'fastfood': 5161, 'fuel': 1406,
                      'taxi': 2742, 'groceries': 17258, 'pharmacies': 1935,
                      'cinema': 3548, 'airlines': 10806, 'other': 19728}
    
    LAYOUT = ipywidgets.Layout(width='auto', height='auto')
    
    def __init__(self):
        self.vector = None
        self.strategies = None
        self.my_card = None
        
        self.category_widgets = []

        for category in CATEGORY_NAMES:
            widget = ipywidgets.IntSlider(min=0, max=100000, value=self.DEFAULT_VALUES[category], description=category)
            self.category_widgets.append(widget)

        self.my_card_widget = ipywidgets.Dropdown(options=CARD_NAMES, value=CARD_NAMES[0], description='Your card:')

        self.top_strategies_button = ipywidgets.Button(description="Run", layout=self.LAYOUT)
        self.top_strategies_output = ipywidgets.Output()

        self.strategy_choice_widget = ipywidgets.RadioButtons(options=np.arange(1, 6), description='Show:', layout=self.LAYOUT)
        self.strategy_choice_button = ipywidgets.Button(description='Show strategy', layout=self.LAYOUT)
        self.strategy_choice_output = ipywidgets.Output()
        
        display(HTML('<h2>Cashback Calculator</h2>'))
        display(HTML('<h4>Your spendings in rubles:</h4>'))
        display(*self.category_widgets, self.my_card_widget, self.top_strategies_button)
        display(self.top_strategies_output, self.strategy_choice_output)

        self.top_strategies_button.on_click(self.display_top_strategies)
        self.strategy_choice_button.on_click(self.display_strategy)
        
    def set_my_card(self):
        my_card_id = CARD_NAMES.index(self.my_card_widget.value) - 1

        if my_card_id >= 0:
            self.my_card = CARDS[my_card_id]
        
    def display_top_strategies(self, button):
        amount_spent_by_category = {
            category: widget.value
            for category, widget in zip(CATEGORY_NAMES, self.category_widgets)
        }
        self.vector = get_vector(amount_spent_by_category)

        self.set_my_card()

        self.top_strategies_output.clear_output()
        self.strategy_choice_output.clear_output()

        with self.top_strategies_output:
            display(HTML('Optimizing the spending strategy.'))

            self.strategies = optimize_cashback(self.vector, CARDS, progress_bar=True)

            display(HTML('Finished.'))

            if self.my_card:
                my_card_profit = self.my_card.calculate_profit(self.vector)
                self.strategies = [strategy for strategy in self.strategies if strategy[-1] > my_card_profit]

                display(HTML(f'Cashback for your card: {round(my_card_profit)} rub.\n'))

            display(HTML('<h4>Best strategies for you:</h4>'))

            self.strategy_choice_widget.options = [
                (f"Cashback: {round(profit)} rub. Cards '{c1.card_name}' and '{c2.card_name}'.", i)
                for i, (c1, c2, _, _, _, profit) in enumerate(self.strategies[:5], start=1)
            ]

            display(self.strategy_choice_widget, self.strategy_choice_button)
            
    def display_strategy(self, button):
        strategy_id = self.strategy_choice_widget.value - 1
        c1, c2, c1_benchmark_id, c2_benchmark_id, x, profit = self.strategies[strategy_id]

        self.set_my_card()

        self.strategy_choice_output.clear_output()

        with self.strategy_choice_output:
            display(HTML(f"<h4>Chosen strategy: cards '{c1.card_name}' and '{c2.card_name}'.</h4>"))

            t0, t1 = c1.benchmarks[[c1_benchmark_id, c1_benchmark_id + 1]]
            s0, s1 = c2.benchmarks[[c2_benchmark_id, c2_benchmark_id + 1]]

            plot_cashback_cases(profit, self.vector, c1, c2, self.my_card)

            strategy_conditions = f'Necessary conditions:<ul> <li>at least {round(t0)} rub. '

            if t1 < np.infty:
                strategy_conditions += f'and up to {round(t1)} rub. '

            strategy_conditions += f'spent with first card,</li> <li>at least {round(s0)} rub. '

            if s1 < np.infty:
                strategy_conditions += f'and up to {round(s1)} rub. '

            strategy_conditions += 'spent with second card.</li></ul>'
            display(HTML(strategy_conditions))

            plot_strategy(self.vector, x, c1, c2)
