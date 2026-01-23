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


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_torch_return_intermediate_weights():
    """
    PyTorch backend should support return_intermediate_weights=True.
    """
    import torch

    rows, cue_matrix, out_matrix = _make_dummy_mats()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w_final = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device=device,
    )

    w_intermediate = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=True,
        backend='torch',
        device=device,
    )

    assert isinstance(w_intermediate, list), 'Should return a list when return_intermediate_weights=True'
    assert len(w_intermediate) == len(rows) + 1, f'Expected {len(rows) + 1} weight matrices, got {len(w_intermediate)}'
    assert np.allclose(
        w_intermediate[-1].values.astype('float32'),
        w_final.values.astype('float32'),
        atol=1e-5,
    ), 'Last intermediate weight should match final weight'


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_torch_cpu_explicit():
    """
    Test torch backend with explicit device='cpu'.
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

    w_torch_cpu = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device='cpu',
    )

    assert np.allclose(
        w_torch_cpu.values.astype('float32'),
        w_numpy.values.astype('float32'),
        atol=1e-5,
    ), 'Torch CPU backend should match NumPy backend'


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_torch_cuda_when_available():
    """
    Test torch backend with CUDA when available.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_cpu = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device='cpu',
    )

    w_cuda = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device='cuda',
    )

    assert np.allclose(
        w_cuda.values.astype('float32'),
        w_cpu.values.astype('float32'),
        atol=1e-5,
    ), 'CUDA and CPU results should match'


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_different_learning_rates():
    """
    Test that different learning rates produce different results.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()

    w_01 = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    w_05 = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.5,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    # Results should be different with different learning rates
    assert not np.allclose(w_01.values, w_05.values), 'Different learning rates should produce different results'

    # Test that torch backend also respects learning rate
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w_torch_05 = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.5,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device=device,
    )

    assert np.allclose(
        w_torch_05.values.astype('float32'),
        w_05.values.astype('float32'),
        atol=1e-5,
    ), 'Torch backend should respect learning rate parameter'


def test_preinitalized_weight_matrix():
    """
    Test incremental learning with a pre-initialized weight matrix.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()

    # Create a pre-initialized weight matrix
    import xarray as xr
    initial_weights = np.random.RandomState(42).normal(size=(cue_matrix.shape[1], out_matrix.shape[1]))
    weight_matrix = xr.DataArray(
        initial_weights,
        dims=('cues', 'semantics'),
        coords={'cues': cue_matrix.cues.values, 'semantics': out_matrix.semantics.values},
    )

    w_from_init = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=weight_matrix,
        return_intermediate_weights=False,
        backend='numpy',
    )

    w_from_zero = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    # Results should be different when starting from different initial weights
    assert not np.allclose(w_from_init.values, w_from_zero.values), 'Pre-initialized weights should affect the result'


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_torch_preinitalized_weight_matrix():
    """
    Test torch backend with a pre-initialized weight matrix.
    """
    import torch
    import xarray as xr

    rows, cue_matrix, out_matrix = _make_dummy_mats()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a pre-initialized weight matrix
    initial_weights = np.random.RandomState(42).normal(size=(cue_matrix.shape[1], out_matrix.shape[1]))
    weight_matrix = xr.DataArray(
        initial_weights,
        dims=('cues', 'semantics'),
        coords={'cues': cue_matrix.cues.values, 'semantics': out_matrix.semantics.values},
    )

    w_numpy = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=weight_matrix.copy(),
        return_intermediate_weights=False,
        backend='numpy',
    )

    w_torch = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=weight_matrix.copy(),
        return_intermediate_weights=False,
        backend='torch',
        device=device,
    )

    assert np.allclose(
        w_torch.values.astype('float32'),
        w_numpy.values.astype('float32'),
        atol=1e-5,
    ), 'Torch and NumPy backends should produce same results with pre-initialized weights'


def test_single_row():
    """
    Test incremental learning with a single row.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()
    single_row = [rows[0]]

    w = lmap.incremental_learning(
        single_row,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    assert w.shape == (cue_matrix.shape[1], out_matrix.shape[1]), 'Output shape should be correct'
    # Weight matrix should not be all zeros after one update
    assert np.any(w.values != 0), 'Weight matrix should be updated'


def test_invalid_backend_raises_error():
    """
    Test that an invalid backend raises ValueError.
    """
    rows, cue_matrix, out_matrix = _make_dummy_mats()

    with pytest.raises(ValueError, match='Unknown backend'):
        lmap.incremental_learning(
            rows,
            cue_matrix,
            out_matrix,
            learning_rate=0.1,
            weight_matrix=None,
            return_intermediate_weights=False,
            backend='invalid_backend',
        )


def test_torch_backend_without_pytorch():
    """
    Test that requesting torch backend without PyTorch raises ImportError.
    """
    # This test is a bit tricky since we need PyTorch to run other tests
    # We'll test the error message is appropriate when torch backend is requested
    # Skip if torch is available
    try:
        import torch
        pytest.skip('PyTorch is installed, cannot test missing PyTorch scenario')
    except ImportError:
        rows, cue_matrix, out_matrix = _make_dummy_mats()
        with pytest.raises(ImportError, match='PyTorch is not installed'):
            lmap.incremental_learning(
                rows,
                cue_matrix,
                out_matrix,
                learning_rate=0.1,
                weight_matrix=None,
                return_intermediate_weights=False,
                backend='torch',
            )


def test_larger_scale_learning():
    """
    Test incremental learning with a larger, more realistic dataset.
    """
    rng = np.random.default_rng(123)

    # Create a larger dataset: 100 words, 50 cues, 20 semantic dimensions
    n_words = 100
    n_cues = 50
    n_semantics = 20

    words = [f'word_{i}' for i in range(n_words)]
    cues = [f'cue_{i}' for i in range(n_cues)]
    semantics = [f'sem_{i}' for i in range(n_semantics)]

    cue_vals = rng.normal(size=(n_words, n_cues)).astype('float32')
    sem_vals = rng.normal(size=(n_words, n_semantics)).astype('float32')

    import xarray as xr
    cue_matrix = xr.DataArray(
        cue_vals,
        dims=('word', 'cues'),
        coords={'word': words, 'cues': cues},
    )
    out_matrix = xr.DataArray(
        sem_vals,
        dims=('word', 'semantics'),
        coords={'word': words, 'semantics': semantics},
    )

    # Test numpy backend
    w_numpy = lmap.incremental_learning(
        words,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='numpy',
    )

    assert w_numpy.shape == (n_cues, n_semantics), 'Output shape should be correct'

    # Test torch backend if available
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        w_torch = lmap.incremental_learning(
            words,
            cue_matrix,
            out_matrix,
            learning_rate=0.1,
            weight_matrix=None,
            return_intermediate_weights=False,
            backend='torch',
            device=device,
        )

        assert np.allclose(
            w_torch.values.astype('float32'),
            w_numpy.values.astype('float32'),
            rtol=1e-4,
            atol=0.1,
        ), 'Torch and NumPy backends should match on larger datasets'
    except ImportError:
        pass  # PyTorch not available, skip torch test


@pytest.mark.skipif(
    pytest.importorskip('torch') is None, reason='PyTorch not installed'
)
def test_auto_backend_prefers_cuda():
    """
    Test that backend='auto' uses CUDA when available.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    rows, cue_matrix, out_matrix = _make_dummy_mats()

    # backend='auto' should use CUDA
    w_auto = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='auto',
    )

    # Should match CUDA result
    w_cuda = lmap.incremental_learning(
        rows,
        cue_matrix,
        out_matrix,
        learning_rate=0.1,
        weight_matrix=None,
        return_intermediate_weights=False,
        backend='torch',
        device='cuda',
    )

    assert np.allclose(
        w_auto.values.astype('float32'),
        w_cuda.values.astype('float32'),
        atol=1e-6,
    ), 'Auto backend should use CUDA when available'

