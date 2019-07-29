from typing import List, Optional

import numpy as np

from myutils.json import load_json

from config import DATA_DIR

__all__ = [
    'ALL_MCCS',
    'N_MCCS',
    'MCC2ID',
    'ID2MCC',
    'MCC_CATEGORIES',
    'CARD_IDS',
    'Card',
]

DB_DIR = DATA_DIR / 'db'

ALL_MCCS_JSON_FPATH = DB_DIR / 'all_mccs.json'
CARDS_JSON_FPATH = DB_DIR / 'cards.json'
BANKS_JSON_FPATH = DB_DIR / 'banks.json'
MCC_CATEGORIES_JSON_FPATH = DB_DIR / 'mcc_categories.json'

ALL_MCCS = load_json(ALL_MCCS_JSON_FPATH)
N_MCCS = len(ALL_MCCS)
MCC2ID = dict(zip(ALL_MCCS, range(N_MCCS)))
ID2MCC = {value: key for key, value in MCC2ID.items()}

BANKS = load_json(BANKS_JSON_FPATH)

CARDS = load_json(CARDS_JSON_FPATH)
CARD_IDS = list(CARDS.keys())

MCC_CATEGORIES = load_json(MCC_CATEGORIES_JSON_FPATH)


class Card:
    def __init__(self, card_id: str, all_mccs: Optional[List[str]] = None) -> None:
        if all_mccs is None:
            self.all_mccs = ALL_MCCS.copy()
        else:
            self.all_mccs = all_mccs

        self.n_mccs = len(self.all_mccs)
        self.mcc2id = dict(zip(self.all_mccs, range(self.n_mccs)))

        self.card_id = card_id
        
        card: dict = CARDS[card_id]
        bank: dict = BANKS[card['bank_id']]
        profit_type = card['profit_type']
        
        self.bank_id = card['bank_id']
        self.card_name = f"{bank['bank_name']} {card['card_name']}"

        self.n_benchmarks = len(card['profit'])
        self.benchmarks = np.zeros(self.n_benchmarks + 1)
        self.benchmarks[-1] = np.infty
        self.profit_rates = np.zeros((self.n_benchmarks, self.n_mccs))
        
        for i, item in enumerate(card['profit']):
            benchmark = item['amount_spent']
            self.benchmarks[i] = benchmark
            
            general_rate = item['general_rate']
            self.profit_rates[i] = general_rate
            
            for category, special_rate in item.get('special_rates', {}).items():
                category_mccs = bank['categories'][profit_type][category]
                
                for mcc in category_mccs:
                    j = self.mcc2id.get(mcc)

                    if j is not None:
                        self.profit_rates[i, j] = special_rate
            
            for mcc in bank['skipped_mccs']:
                j = self.mcc2id.get(mcc)

                if j is not None:
                    self.profit_rates[i, j] = 0
                
        service_cost = card['service_cost']
        
        self.service_cost = service_cost['cost']
        self.free_service_amount_spent = service_cost.get('amount_spent', np.inf)
        self.free_service_balance = service_cost.get('min_balance', np.inf)

    def get_benchmark_id(self, spending_sum: float) -> int:
        if spending_sum < 0:
            spending_sum = 0
        
        return np.where(self.benchmarks <= spending_sum)[0][-1]
        
    def get_profit_rate_vector(self, spending_sum: float) -> np.ndarray:
        i = self.get_benchmark_id(spending_sum)
        
        return self.profit_rates[i]
    
    def get_service_cost(self, spending_sum: float, balance: float = 0) -> float:
        if (spending_sum >= self.free_service_amount_spent) or (balance >= self.free_service_balance):
            return 0
        else:
            return self.service_cost
                
    def calculate_profit(self, spending_vector: np.ndarray, balance: float = 0) -> float:
        spending_sum = spending_vector.sum()
        
        profit_rate = self.get_profit_rate_vector(spending_sum)
        service_cost = self.get_service_cost(spending_sum, balance)
        
        return profit_rate.dot(spending_vector)  # - service_cost
    
    def __repr__(self) -> str:
        return f'Card(card_id={repr(self.card_id)}, bank_id={repr(self.bank_id)})'
