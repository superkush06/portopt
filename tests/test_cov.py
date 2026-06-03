"""Covariance tests."""

import numpy as np

from portopt.cov import ledoit_wolf, sample_cov


def test_sample_cov_shape_and_symmetric():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    S = sample_cov(X)
    assert S.shape == (5, 5)
    np.testing.assert_allclose(S, S.T)


def test_ledoit_wolf_shrinkage_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 30))   # T small relative to N
    S, delta = ledoit_wolf(X)
    assert 0.0 <= delta <= 1.0
    assert S.shape == (30, 30)


def test_ledoit_wolf_less_shrinkage_when_structure_present():
    """When true Σ has real factor structure, LW correctly shrinks less
    than for structureless IID data. (Counterintuitive but right: LW
    shrinks toward scaled identity; if the truth IS identity-like,
    shrinking 100% is optimal; if it has factor structure, the sample
    captures real info that's worth keeping.)
    """
    rng = np.random.default_rng(0)
    # Structureless: IID standard normal
    _, d_iid = ledoit_wolf(rng.standard_normal((500, 20)))

    # Structured: 5 latent factors with strong loadings
    loadings = rng.standard_normal((20, 5))
    F = rng.standard_normal((500, 5))
    idio = 0.1 * rng.standard_normal((500, 20))
    X_struct = F @ loadings.T + idio
    _, d_struct = ledoit_wolf(X_struct)

    assert d_struct < d_iid


def test_ledoit_wolf_psd():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8))
    S, _ = ledoit_wolf(X)
    eigvals = np.linalg.eigvalsh(S)
    assert (eigvals > -1e-10).all()
