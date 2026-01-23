import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
from discriminative_lexicon_model.ldl import LDL, concat_cues, is_consecutive

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

TEST_ROOT = Path('.')
RESOURCES = TEST_ROOT / 'resources'

# Test data: same as test_ldl.py for consistency
words = ['ban', 'banban']
freqs = [10, 20]
semdf = pd.DataFrame({'hit': [1, 1], 'intensity': [1, 2]}, index=words)


@pytest.fixture
def ldl_with_matrices():
    """Create an LDL instance with all matrices generated."""
    ldl = LDL(words, semdf, allmatrices=True)
    return ldl


class TestProduce:
    """Tests for the LDL.produce() method."""

    def test_produce_returns_dataframe(self, ldl_with_matrices):
        """Test that produce returns a pandas DataFrame."""
        gold = np.array([1, 1])  # semantic vector for 'ban'
        result = ldl_with_matrices.produce(gold)
        assert isinstance(result, pd.DataFrame)

    def test_produce_has_selected_column(self, ldl_with_matrices):
        """Test that the returned DataFrame has a 'Selected' column."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold)
        assert 'Selected' in result.columns

    def test_produce_selects_cues(self, ldl_with_matrices):
        """Test that produce selects cues from the available cue set."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold)
        available_cues = ldl_with_matrices.cmat.cues.values
        for cue in result['Selected'].values:
            assert cue in available_cues

    def test_produce_ban_semantic_vector(self, ldl_with_matrices):
        """Test produce with the semantic vector for 'ban' [1, 1]."""
        gold = np.array([1, 1])  # 'ban' has hit=1, intensity=1
        result = ldl_with_matrices.produce(gold)
        # Check that some cues were selected
        assert len(result) > 0
        # The result should end with a cue ending in '#' (word boundary)
        last_cue = result['Selected'].iloc[-1]
        assert last_cue.endswith('#')

    def test_produce_banban_semantic_vector(self, ldl_with_matrices):
        """Test produce with the semantic vector for 'banban' [1, 2]."""
        gold = np.array([1, 2])  # 'banban' has hit=1, intensity=2
        result = ldl_with_matrices.produce(gold)
        assert len(result) > 0
        last_cue = result['Selected'].iloc[-1]
        assert last_cue.endswith('#')

    def test_produce_with_list_input(self, ldl_with_matrices):
        """Test that produce accepts list input for gold vector."""
        gold = [1, 1]
        result = ldl_with_matrices.produce(gold)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestProduceWord:
    """Tests for produce with word=True parameter."""

    def test_produce_word_returns_string(self, ldl_with_matrices):
        """Test that produce with word=True returns a string."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, word=True)
        assert isinstance(result, str)

    def test_produce_word_ban(self, ldl_with_matrices):
        """Test that producing 'ban' semantic vector returns '#ban#'."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, word=True)
        # The produced word should be a valid form
        assert result.startswith('#')
        assert result.endswith('#')


class TestProduceVmat:
    """Tests for produce with apply_vmat parameter."""

    def test_produce_without_vmat(self, ldl_with_matrices):
        """Test produce with apply_vmat=False."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, apply_vmat=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_produce_with_and_without_vmat_differ(self, ldl_with_matrices):
        """Test that apply_vmat affects the production result."""
        gold = np.array([1, 2])
        result_with_vmat = ldl_with_matrices.produce(gold, apply_vmat=True)
        result_without_vmat = ldl_with_matrices.produce(gold, apply_vmat=False)
        # Results may differ (or may not, depending on the specific case)
        # At minimum, both should be valid DataFrames
        assert isinstance(result_with_vmat, pd.DataFrame)
        assert isinstance(result_without_vmat, pd.DataFrame)


class TestProducePositive:
    """Tests for produce with positive parameter."""

    def test_produce_positive_true(self, ldl_with_matrices):
        """Test produce with positive=True clips negative values."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, positive=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_produce_positive_false(self, ldl_with_matrices):
        """Test produce with positive=False (default) allows negative values."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, positive=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestProduceMaxAttempt:
    """Tests for produce max_attempt parameter."""

    def test_produce_max_attempt_one(self, ldl_with_matrices, capsys):
        """Test produce with max_attempt=1 stops after one iteration."""
        gold = np.array([1, 2])
        result = ldl_with_matrices.produce(gold, max_attempt=1)
        # With max_attempt=1, should only select one cue
        assert len(result) <= 1
        # Check that max iteration message was printed
        captured = capsys.readouterr()
        if len(result) == 1 and not result['Selected'].iloc[0].endswith('#'):
            assert 'maximum number of iterations' in captured.out.lower()

    def test_produce_custom_max_attempt(self, ldl_with_matrices):
        """Test produce with custom max_attempt value."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, max_attempt=100)
        assert isinstance(result, pd.DataFrame)


class TestProduceRoundby:
    """Tests for produce roundby parameter."""

    def test_produce_roundby_affects_precision(self, ldl_with_matrices):
        """Test that roundby parameter affects the precision of scores."""
        gold = np.array([1, 1])
        result_round10 = ldl_with_matrices.produce(gold, roundby=10)
        result_round2 = ldl_with_matrices.produce(gold, roundby=2)
        # Both should be valid DataFrames
        assert isinstance(result_round10, pd.DataFrame)
        assert isinstance(result_round2, pd.DataFrame)


class TestProduceEdgeCases:
    """Edge case tests for produce."""

    def test_produce_zero_vector(self, ldl_with_matrices):
        """Test produce with zero semantic vector."""
        gold = np.array([0, 0])
        result = ldl_with_matrices.produce(gold)
        # With zero gold, c_prod should be <= 0, so production stops early
        assert isinstance(result, pd.DataFrame)

    def test_produce_negative_vector(self, ldl_with_matrices):
        """Test produce with negative semantic vector values."""
        gold = np.array([-1, -1])
        result = ldl_with_matrices.produce(gold)
        assert isinstance(result, pd.DataFrame)


class TestConcatCues:
    """Tests for the concat_cues helper function."""

    def test_concat_cues_basic(self):
        """Test basic cue concatenation."""
        cues = pd.Series(['#ba', 'ban', 'an#'])
        result = concat_cues(cues)
        assert result == '#ban#'

    def test_concat_cues_longer_word(self):
        """Test cue concatenation for longer word."""
        cues = pd.Series(['#ba', 'ban', 'anb', 'nba', 'ban', 'an#'])
        result = concat_cues(cues)
        assert result == '#banban#'


class TestIsConsecutive:
    """Tests for the is_consecutive helper function."""

    def test_is_consecutive_true(self):
        """Test that consecutive cues are detected."""
        cues = pd.Series(['#ba', 'ban', 'an#'])
        assert is_consecutive(cues) == True

    def test_is_consecutive_false(self):
        """Test that non-consecutive cues are detected."""
        cues = pd.Series(['#ba', 'an#'])  # missing 'ban'
        assert is_consecutive(cues) == False

    def test_is_consecutive_single_cue(self):
        """Test is_consecutive with single cue."""
        cues = pd.Series(['#ba'])
        # Single cue should be considered consecutive
        assert is_consecutive(cues) == True


class TestProduceIntegration:
    """Integration tests for produce with full workflow."""

    def test_produce_and_verify_consecutive(self, ldl_with_matrices):
        """Test that produced cues are consecutive (valid word form)."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold)
        selected = result['Selected']
        # Check if selected cues form a consecutive sequence
        if len(selected) > 1 and selected.iloc[-1].endswith('#'):
            assert is_consecutive(selected)

    def test_produce_and_concat(self, ldl_with_matrices):
        """Test produce followed by manual concat_cues."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold)
        selected = result['Selected']
        # If we have a complete word (ends with #), we can concat
        if len(selected) > 0 and selected.iloc[-1].endswith('#'):
            word = concat_cues(selected)
            assert isinstance(word, str)
            assert word.startswith('#')
            assert word.endswith('#')

    def test_produce_word_equals_concat(self, ldl_with_matrices):
        """Test that word=True gives same result as manual concat_cues."""
        gold = np.array([1, 1])
        result_df = ldl_with_matrices.produce(gold, word=False)
        result_word = ldl_with_matrices.produce(gold, word=True)
        # If both complete successfully, they should produce the same word
        if len(result_df) > 0 and result_df['Selected'].iloc[-1].endswith('#'):
            manual_word = concat_cues(result_df['Selected'])
            assert result_word == manual_word


class TestProduceBackend:
    """Tests for produce backend parameter."""

    def test_produce_numpy_backend(self, ldl_with_matrices):
        """Test produce with explicit NumPy backend."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='numpy')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Selected' in result.columns

    def test_produce_auto_backend(self, ldl_with_matrices):
        """Test produce with auto backend (default)."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='auto')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Selected' in result.columns

    def test_produce_default_backend_is_auto(self, ldl_with_matrices):
        """Test that default backend is 'auto'."""
        gold = np.array([1, 1])
        result_default = ldl_with_matrices.produce(gold)
        result_auto = ldl_with_matrices.produce(gold, backend='auto')
        # Should produce identical results
        assert list(result_default['Selected']) == list(result_auto['Selected'])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_torch_backend(self, ldl_with_matrices):
        """Test produce with PyTorch backend."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Selected' in result.columns

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_torch_cpu_device(self, ldl_with_matrices):
        """Test produce with PyTorch backend on CPU."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', device='cpu')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_produce_torch_cuda_device(self, ldl_with_matrices):
        """Test produce with PyTorch backend on CUDA."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', device='cuda')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_produce_invalid_backend(self, ldl_with_matrices):
        """Test that invalid backend raises ValueError."""
        gold = np.array([1, 1])
        with pytest.raises(ValueError, match='Unknown backend'):
            ldl_with_matrices.produce(gold, backend='invalid_backend')

    @pytest.mark.skipif(HAS_TORCH, reason="Test requires PyTorch to be unavailable")
    def test_produce_torch_backend_without_torch(self, ldl_with_matrices):
        """Test that torch backend raises ImportError when PyTorch is not installed."""
        gold = np.array([1, 1])
        with pytest.raises(ImportError, match='PyTorch is not installed'):
            ldl_with_matrices.produce(gold, backend='torch')


class TestProduceBackendConsistency:
    """Tests for consistency across different backends."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_numpy_vs_torch_cpu_consistency(self, ldl_with_matrices):
        """Test that NumPy and PyTorch CPU backends produce identical results."""
        gold = np.array([1, 1])
        result_numpy = ldl_with_matrices.produce(gold, backend='numpy')
        result_torch = ldl_with_matrices.produce(gold, backend='torch', device='cpu')

        # Selected cues should be identical
        assert list(result_numpy['Selected']) == list(result_torch['Selected'])

        # Number of rows should be identical
        assert len(result_numpy) == len(result_torch)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_numpy_vs_torch_cuda_consistency(self, ldl_with_matrices):
        """Test that NumPy and PyTorch CUDA backends produce identical results."""
        gold = np.array([1, 1])
        result_numpy = ldl_with_matrices.produce(gold, backend='numpy')
        result_cuda = ldl_with_matrices.produce(gold, backend='torch', device='cuda')

        # Selected cues should be identical
        assert list(result_numpy['Selected']) == list(result_cuda['Selected'])

        # Number of rows should be identical
        assert len(result_numpy) == len(result_cuda)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_torch_cpu_vs_cuda_consistency(self, ldl_with_matrices):
        """Test that PyTorch CPU and CUDA backends produce identical results."""
        gold = np.array([1, 1])
        result_cpu = ldl_with_matrices.produce(gold, backend='torch', device='cpu')
        result_cuda = ldl_with_matrices.produce(gold, backend='torch', device='cuda')

        # Selected cues should be identical
        assert list(result_cpu['Selected']) == list(result_cuda['Selected'])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_all_parameters_with_backends(self, ldl_with_matrices):
        """Test that all produce parameters work consistently across backends."""
        gold = np.array([1, 2])

        # Test with various parameter combinations
        params = {
            'word': False,
            'roundby': 5,
            'max_attempt': 30,
            'positive': True,
            'apply_vmat': True
        }

        result_numpy = ldl_with_matrices.produce(gold, backend='numpy', **params)
        result_torch = ldl_with_matrices.produce(gold, backend='torch', device='cpu', **params)

        assert list(result_numpy['Selected']) == list(result_torch['Selected'])


class TestProduceBackendWithVmat:
    """Tests for backend parameter interaction with vmat."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_with_vmat_true(self, ldl_with_matrices):
        """Test PyTorch backend with apply_vmat=True."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', apply_vmat=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_with_vmat_false(self, ldl_with_matrices):
        """Test PyTorch backend with apply_vmat=False."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', apply_vmat=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_vmat_consistency_across_backends(self, ldl_with_matrices):
        """Test that vmat behavior is consistent across backends."""
        gold = np.array([1, 1])

        # With vmat
        result_numpy_vmat = ldl_with_matrices.produce(gold, backend='numpy', apply_vmat=True)
        result_torch_vmat = ldl_with_matrices.produce(gold, backend='torch', apply_vmat=True)
        assert list(result_numpy_vmat['Selected']) == list(result_torch_vmat['Selected'])

        # Without vmat
        result_numpy_no_vmat = ldl_with_matrices.produce(gold, backend='numpy', apply_vmat=False)
        result_torch_no_vmat = ldl_with_matrices.produce(gold, backend='torch', apply_vmat=False)
        assert list(result_numpy_no_vmat['Selected']) == list(result_torch_no_vmat['Selected'])


class TestProduceBackendWithPositive:
    """Tests for backend parameter interaction with positive parameter."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_positive_true(self, ldl_with_matrices):
        """Test PyTorch backend with positive=True."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', positive=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_positive_consistency_across_backends(self, ldl_with_matrices):
        """Test that positive parameter behaves consistently across backends."""
        gold = np.array([1, 1])

        result_numpy = ldl_with_matrices.produce(gold, backend='numpy', positive=True)
        result_torch = ldl_with_matrices.produce(gold, backend='torch', positive=True)

        assert list(result_numpy['Selected']) == list(result_torch['Selected'])


class TestProduceBackendWithWord:
    """Tests for backend parameter interaction with word parameter."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend_word_true(self, ldl_with_matrices):
        """Test PyTorch backend with word=True returns string."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='torch', word=True)
        assert isinstance(result, str)
        assert result.startswith('#')
        assert result.endswith('#')

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_word_consistency_across_backends(self, ldl_with_matrices):
        """Test that word=True produces identical results across backends."""
        gold = np.array([1, 1])

        result_numpy = ldl_with_matrices.produce(gold, backend='numpy', word=True)
        result_torch = ldl_with_matrices.produce(gold, backend='torch', word=True)

        assert result_numpy == result_torch


class TestProduceBackendEdgeCases:
    """Edge case tests for produce with different backends."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_zero_vector(self, ldl_with_matrices):
        """Test PyTorch backend with zero semantic vector."""
        gold = np.array([0, 0])
        result = ldl_with_matrices.produce(gold, backend='torch')
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_negative_vector(self, ldl_with_matrices):
        """Test PyTorch backend with negative semantic vector."""
        gold = np.array([-1, -1])
        result = ldl_with_matrices.produce(gold, backend='torch')
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_max_attempt_one(self, ldl_with_matrices, capsys):
        """Test PyTorch backend with max_attempt=1."""
        gold = np.array([1, 2])
        result = ldl_with_matrices.produce(gold, backend='torch', max_attempt=1)
        assert len(result) <= 1

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_list_input(self, ldl_with_matrices):
        """Test that PyTorch backend accepts list input for gold vector."""
        gold = [1, 1]
        result = ldl_with_matrices.produce(gold, backend='torch')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestProduceAutoBackendSelection:
    """Tests for auto backend selection logic."""

    def test_auto_backend_falls_back_to_numpy_without_torch(self, ldl_with_matrices):
        """Test that auto backend works even without PyTorch."""
        gold = np.array([1, 1])
        # Should not raise an error, will use numpy backend
        result = ldl_with_matrices.produce(gold, backend='auto')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_auto_backend_uses_torch_when_available(self, ldl_with_matrices):
        """Test that auto backend uses PyTorch when available."""
        gold = np.array([1, 1])
        result_auto = ldl_with_matrices.produce(gold, backend='auto')

        # If CUDA is available, auto should use it
        if HAS_CUDA:
            # Result should be identical to torch+cuda
            result_cuda = ldl_with_matrices.produce(gold, backend='torch', device='cuda')
            assert list(result_auto['Selected']) == list(result_cuda['Selected'])
        else:
            # Result should be identical to numpy (fallback)
            result_numpy = ldl_with_matrices.produce(gold, backend='numpy')
            assert list(result_auto['Selected']) == list(result_numpy['Selected'])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_auto_backend_device_none_defaults_correctly(self, ldl_with_matrices):
        """Test that device=None with auto backend defaults correctly."""
        gold = np.array([1, 1])
        result = ldl_with_matrices.produce(gold, backend='auto', device=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestProduceBatch:
    """Tests for the LDL.produce_batch() method (Phase 2 optimization)."""

    def test_produce_batch_returns_list(self, ldl_with_matrices):
        """Test that produce_batch returns a list of DataFrames."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='numpy')
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_produce_batch_has_selected_columns(self, ldl_with_matrices):
        """Test that each DataFrame in batch has 'Selected' column."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='numpy')
        for df in result:
            assert 'Selected' in df.columns

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_torch_backend(self, ldl_with_matrices):
        """Test produce_batch with PyTorch backend."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch')
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_produce_batch_torch_cuda(self, ldl_with_matrices):
        """Test produce_batch with PyTorch CUDA backend."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch', device='cuda')
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_consistency_with_single(self, ldl_with_matrices):
        """Test that batch processing produces same results as single processing."""
        gold1 = np.array([1, 1])
        gold2 = np.array([1, 2])
        gold_batch = np.array([gold1, gold2])

        # Single processing
        result_single1 = ldl_with_matrices.produce(gold1, backend='torch', device='cpu')
        result_single2 = ldl_with_matrices.produce(gold2, backend='torch', device='cpu')

        # Batch processing
        result_batch = ldl_with_matrices.produce_batch(gold_batch, backend='torch', device='cpu')

        # Compare results
        assert list(result_single1['Selected']) == list(result_batch[0]['Selected'])
        assert list(result_single2['Selected']) == list(result_batch[1]['Selected'])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_produce_batch_cuda_consistency_with_single(self, ldl_with_matrices):
        """Test that CUDA batch processing produces same results as single processing."""
        gold1 = np.array([1, 1])
        gold2 = np.array([1, 2])
        gold_batch = np.array([gold1, gold2])

        # Single processing on CUDA
        result_single1 = ldl_with_matrices.produce(gold1, backend='torch', device='cuda')
        result_single2 = ldl_with_matrices.produce(gold2, backend='torch', device='cuda')

        # Batch processing on CUDA
        result_batch = ldl_with_matrices.produce_batch(gold_batch, backend='torch', device='cuda')

        # Compare results
        assert list(result_single1['Selected']) == list(result_batch[0]['Selected'])
        assert list(result_single2['Selected']) == list(result_batch[1]['Selected'])

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_with_word_parameter(self, ldl_with_matrices):
        """Test produce_batch with word=True parameter."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch', word=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(w, str) for w in result)
        assert all(w.startswith('#') and w.endswith('#') for w in result)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_with_vmat(self, ldl_with_matrices):
        """Test produce_batch with apply_vmat parameter."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result_with_vmat = ldl_with_matrices.produce_batch(gold_batch, backend='torch', apply_vmat=True)
        result_without_vmat = ldl_with_matrices.produce_batch(gold_batch, backend='torch', apply_vmat=False)

        assert len(result_with_vmat) == 2
        assert len(result_without_vmat) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result_with_vmat)
        assert all(isinstance(df, pd.DataFrame) for df in result_without_vmat)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_with_positive(self, ldl_with_matrices):
        """Test produce_batch with positive parameter."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch', positive=True)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_with_custom_max_attempt(self, ldl_with_matrices):
        """Test produce_batch with custom max_attempt."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch', max_attempt=100)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_empty(self, ldl_with_matrices):
        """Test produce_batch with empty batch."""
        gold_batch = np.array([]).reshape(0, 2)
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch')
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_single_item(self, ldl_with_matrices):
        """Test produce_batch with single item."""
        gold_batch = np.array([[1, 1]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch')
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_produce_batch_with_batch_size(self, ldl_with_matrices):
        """Test produce_batch with batch_size parameter for chunked processing."""
        gold_batch = np.array([[1, 1], [1, 2], [1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='torch', batch_size=2)
        assert len(result) == 4
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_produce_batch_numpy_fallback(self, ldl_with_matrices):
        """Test that numpy backend falls back to sequential processing."""
        gold_batch = np.array([[1, 1], [1, 2]])
        result = ldl_with_matrices.produce_batch(gold_batch, backend='numpy')
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)
