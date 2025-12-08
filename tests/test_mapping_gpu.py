import numpy as np
import xarray as xr
import pytest

from discriminative_lexicon_model import mapping as lmap


def _make_dummy_mats():
    """
    Small deterministic matrices for testing incremental_learning.
    """

    rng = np.random.default_rng(518)

    words = ['w1', 'w2', 'w3', 'w4']
    cues = ['c1', 'c2', 'c3']
    outs = ['o1', 'o2']

    cue_vals = rng.normal(size=(len(words), len(cues))).astype('float32')
    out_vals = rng.normal(size=(len(words), len(outs))).astype('float32')

    cue_matrix = xr.DataArray(
        cue_vals,
        dims=('word', 'cues'),
        coords={'word': words, 'cues': cues},
    )
    out_matrix = xr.DataArray(
        out_vals,
        dims=('word', 'semantics'),
        coords={'word': words, 'semantics': outs},
    )

    rows = words
    return rows, cue_matrix, out_matrix

def test_numpy_backend_matches_private():
    """
    Incremental_learning with backend='numpy'.
    """

    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_private = lmap._incremental_learning_numpy(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
    )

    w_public = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    assert np.allclose(
        w_private.values, w_public.values
    ), (
        'The original implementation "_incremental_learning_numpy" and the new '
        'implementation "incremental_learning" produce different results with '
        'backend="numpy".'
    )


def test_numpy_return_intermediate_weights():
    """
    Return a list of weight matrices with correct length and final value.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_final = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    w_intermediate = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=True,
        backend='numpy',
    )

    assert isinstance(w_intermediate, list)
    assert len(w_intermediate) == len(rows) + 1
    assert np.allclose(
        w_intermediate[-1].values, w_final.values
    ), (
        'The last state of the weight matrix when '
        'return_intermediate_weights=True differs from the final state of the '
        'weight matrix when return_intermediate_weights=False.'
    )


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_torch_backend_matches_numpy():
    """
    Torch backend (CUDA if available, else CPU) should match NumPy within tolerance.
    """
    import torch

    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_numpy = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    # Prefer CUDA; fall back to CPU if no GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w_torch = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device=device,
    )

    np_numpy = w_numpy.values.astype('float32')
    np_torch = w_torch.values.astype('float32')

    assert np.allclose(
        np_torch, np_numpy, atol=1e-5
    ), f'Torch backend on {device} diverges from NumPy backend'


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_auto_backend_consistent_with_numpy():
    """
    backend='auto' should produce the same result as a pure NumPy run.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_numpy = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    w_auto = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='auto',
    )

    assert np.allclose(
        w_auto.values.astype('float32'),
        w_numpy.values.astype('float32'),
        atol=1e-5,
    ), 'backend="auto" result does not match NumPy backend'

