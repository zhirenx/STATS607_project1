"""Statistical validation of the two inferential tools used in the results.

Simulations use a fixed seed, so these tests are deterministic.
"""

import numpy as np

from src.analysis import metrics

N_SENTENCES = 872  # size of the SST-2 validation set
REPLICATES = 4000
ALPHA = 0.05


def rejection_rate(probs: list[float], seed: int) -> float:
    """Simulate paired outcomes and return how often McNemar's test rejects at ALPHA.

    ``probs`` gives the chance that a sentence is classified correctly by
    both models, by A only, by B only, and by neither.
    """
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(N_SENTENCES, probs, size=REPLICATES)
    p_values = [metrics.mcnemar_exact_p(int(a), int(b)) for _, a, b, _ in counts]
    return float(np.mean(np.array(p_values) < ALPHA))


def test_mcnemar_controls_type_one_error():
    # Two models with equal accuracy (93%) that disagree on 6% of sentences,
    # roughly like BERT and RoBERTa here. The exact test should reject a true
    # null at most ALPHA of the time, allowing three Monte Carlo standard errors.
    rate = rejection_rate([0.90, 0.03, 0.03, 0.04], seed=607)
    standard_error = np.sqrt(ALPHA * (1 - ALPHA) / REPLICATES)
    assert rate <= ALPHA + 3 * standard_error


def test_mcnemar_detects_a_real_difference():
    # A gets 3 points more sentences right than B: the test should usually see it.
    assert rejection_rate([0.88, 0.045, 0.015, 0.06], seed=507) > 0.8


def test_wilson_interval_covers_the_true_accuracy():
    rng = np.random.default_rng(2026)
    true_accuracy = 0.93
    successes = rng.binomial(N_SENTENCES, true_accuracy, size=REPLICATES)
    intervals = [metrics.wilson_interval(int(k), N_SENTENCES) for k in successes]
    coverage = np.mean([low <= true_accuracy <= high for low, high in intervals])
    assert 0.93 <= coverage <= 0.97
