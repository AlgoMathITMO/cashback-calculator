from typing import Tuple, List
from operator import itemgetter
from itertools import combinations

import numpy as np
from scipy.optimize import linprog
from tqdm.notebook import tqdm

from cashback_calculator.db import Card

__all__ = [
    'optimize_cashback_at_benchmarks',
    'optimize_cashback_for_cards',
    'optimize_cashback',
]

ERR = 10 ** (-8)


def optimize_cashback_at_benchmarks(
        spendings_vector: np.ndarray,
        card1: Card,
        card2: Card,
        i1: int,
        i2: int,
) -> Tuple[np.ndarray, float]:
    """Optimize cashback for the given cards and within the given benchmarks.

    The linear programming task is solved using the Mosek Interior Point
    Optimizer (see `https://link.springer.com/chapter/10.1007/978-1-4757-3216-0_8`
    and `scipy.optimize.linprog`).

    Returns:
        * the point at which the maximum is reached (1d array),
        * the corresponding profit (float).

    :param spendings_vector: spendings vector (1d array)
    :param card1: card 1 (Card)
    :param card2: card 2 (Card)
    :param i1: benchmark index for card 1 (int)
    :param i2: benchmark index for card 2 (int)
    :return: tuple
    """

    dim = spendings_vector.size

    total = spendings_vector.sum()

    t0, t1 = card1.benchmarks[[i1, i1 + 1]]
    s0, s1 = card2.benchmarks[[i2, i2 + 1]]

    if not (t0 + s0 <= total <= t1 + s1):
        return np.zeros(dim) * np.nan, -np.infty

    benchmark_border_normals = np.array([-spendings_vector, spendings_vector])

    lower_benchmark_border_intercept = max(t0, total - s1)
    upper_benchmark_border_intercept = min(t1, total - s0)

    benchmark_border_intercepts = np.array([-lower_benchmark_border_intercept,
                                            upper_benchmark_border_intercept])

    c = card1.profit_rates[i1] * spendings_vector
    d = card2.profit_rates[i2] * spendings_vector

    slope = c - d

    if np.linalg.norm(slope) < ERR:
        x = np.ones(dim)
        profit = card1.calculate_profit(spendings_vector)

        return x, profit

    result = linprog(
        -slope,
        A_ub=benchmark_border_normals,
        b_ub=benchmark_border_intercepts,
        bounds=(0, 1),
        method='interior-point',
    )

    x = result.x
    profit = c.dot(x) + d.dot(1 - x)

    return x, profit


def optimize_cashback_for_cards(
        spendings_vector: np.ndarray,
        card1: Card,
        card2: Card,
) -> Tuple[int, int, np.ndarray, float]:
    """Optimize cashback for the given cards.

    Returns:
        * optimal benchmark index for card 1 (int),
        * optimal benchmark index for card 2 (int),
        * the point at which the maximum is reached (1d array),
        * the corresponding profit (float).

    :param spendings_vector: spendings vector (1d array)
    :param card1: card 1 (Card)
    :param card2: card 2 (Card)
    :return: tuple
    """

    results = []

    for i1 in range(card1.n_benchmarks):
        for i2 in range(card2.n_benchmarks):
            x, profit = optimize_cashback_at_benchmarks(
                spendings_vector=spendings_vector,
                card1=card1,
                card2=card2,
                i1=i1,
                i2=i2,
            )

            results.append((i1, i2, x, profit))

    i1, i2, x, profit = max(results, key=itemgetter(3))

    return i1, i2, x, profit


def optimize_cashback(
        spendings_vector: np.ndarray,
        cards: List[Card],
        progress_bar: bool = False,
) -> List[Tuple[Card, Card, int, int, np.ndarray, float]]:
    """Optimize cashback across the given cards.

    Returns a list of tuples containing
        * card 1 (Card),
        * card 2 (Card),
        * optimal benchmark index for card 1 (int),
        * optimal benchmark index for card 2 (int),
        * the point at which the maximum is reached (1d array),
        * the corresponding profit (float).

    This list of sorted in descending order of the corresponding
    profit.

    :param spendings_vector: spendings vector (1d array)
    :param cards: list of Card
    :param progress_bar: display progress bar (bool)
    :return: tuple
    """

    n_cards = len(cards)
    n_iter = n_cards * (n_cards - 1) // 2
    results = []

    for card1, card2 in tqdm(combinations(cards, r=2), total=n_iter, disable=not progress_bar):
        i1, i2, x, profit = optimize_cashback_for_cards(spendings_vector, card1, card2)
        results.append((card1, card2, i1, i2, x, profit))

    results.sort(key=itemgetter(5), reverse=True)

    return results
