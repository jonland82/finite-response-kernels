import random

import numpy as np

from run_constraint_comparison import (
    ambiguity_count_at_d_minus_2,
    constrained_decode,
    exact_boundary_decode,
    fixed_lag_schema,
    validate_sharp_boundary,
)
from run_experiment import tree_profile


def test_sharp_boundary_validation() -> None:
    validation = validate_sharp_boundary()
    assert validation["ambiguous_prefix_equal"]
    assert validation["boundary_decodes_exactly"]
    assert min(validation["version_counts"]) > 1


def test_constructive_decoder_recovers_all_families() -> None:
    rng = random.Random(12)
    for degree in (8, 12, 16):
        for family in ("finite_horizon", "slow", "two_scale", "random_split"):
            schema = fixed_lag_schema(rng, family, degree)
            assert exact_boundary_decode(schema, degree) == schema


def test_integer_decoder_recovers_noiseless_boundary() -> None:
    degree = 8
    schema = fixed_lag_schema(random.Random(7), "finite_horizon", degree)
    observed = tree_profile(schema, degree)[: degree - 1]
    estimate, status = constrained_decode(observed, degree, 2.0)
    truth = np.asarray([schema.get(lag, 0) for lag in range(1, degree + 1)])
    assert status == "solved"
    assert np.array_equal(estimate, truth)


def test_d_minus_2_has_multiple_versions() -> None:
    degree = 10
    for parameter in (1, 17, 100):
        schema = {
            degree - 2: parameter,
            degree - 1: (1 << (degree - 1)) - 3 * parameter,
            degree: 2 * parameter,
        }
        assert ambiguity_count_at_d_minus_2(schema, degree) > 1
