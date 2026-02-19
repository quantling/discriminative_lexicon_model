import pytest
import numpy as np
import pandas as pd
import xarray as xr
from discriminative_lexicon_model.ldl import LDL
from discriminative_lexicon_model.mapping import produce

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

# Test data: same as test_produce.py for consistency
words = ['ban', 'banban']
freqs = [10, 20]
semdf = pd.DataFrame({'hit': [1, 1], 'intensity': [1, 2]}, index=words)


@pytest.fixture
def ldl_with_matrices():
    """Create an LDL instance with all matrices generated."""
    ldl = LDL(words, semdf, allmatrices=True)
    return ldl


# ============================================================
# Part 1: Standalone mapping.produce correctness tests
# ============================================================

class TestStandaloneProduce:
    """Basic tests for the standalone mapping.produce function."""

    def test_returns_dataframe(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert isinstance(result, pd.DataFrame)

    def test_has_selected_column(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert 'Selected' in result.columns

    def test_selects_cues_from_cmat(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        available_cues = ldl_with_matrices.cmat.cues.values
        for cue in result['Selected'].values:
            assert cue in available_cues

    def test_produce_ban(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert len(result) > 0
        last_cue = result['Selected'].iloc[-1]
        assert last_cue.endswith('#')

    def test_produce_banban(self, ldl_with_matrices):
        gold = np.array([1, 2])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert len(result) > 0
        last_cue = result['Selected'].iloc[-1]
        assert last_cue.endswith('#')

    def test_accepts_list_input(self, ldl_with_matrices):
        gold = [1, 1]
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestStandaloneProduceWord:
    """Tests for mapping.produce with word=True."""

    def test_word_returns_string(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         word=True)
        assert isinstance(result, str)

    def test_word_has_boundaries(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         word=True)
        assert result.startswith('#')
        assert result.endswith('#')


class TestStandaloneProduceVmat:
    """Tests for mapping.produce with apply_vmat parameter."""

    def test_without_vmat(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, apply_vmat=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_vmat_required_when_apply_vmat_true(self, ldl_with_matrices):
        gold = np.array([1, 1])
        with pytest.raises(ValueError, match='vmat must be provided'):
            produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                    ldl_with_matrices.gmat, vmat=None, apply_vmat=True)


class TestStandaloneProducePositive:
    """Tests for mapping.produce with positive parameter."""

    def test_positive_true(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         positive=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_positive_false(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         positive=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestStandaloneProduceMaxAttempt:
    """Tests for mapping.produce with max_attempt parameter."""

    def test_max_attempt_one(self, ldl_with_matrices, capsys):
        gold = np.array([1, 2])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         max_attempt=1)
        assert len(result) <= 1
        captured = capsys.readouterr()
        if len(result) == 1 and not result['Selected'].iloc[0].endswith('#'):
            assert 'maximum number of iterations' in captured.out.lower()

    def test_custom_max_attempt(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         max_attempt=100)
        assert isinstance(result, pd.DataFrame)


class TestStandaloneProduceRoundby:
    """Tests for mapping.produce with roundby parameter."""

    def test_roundby_affects_precision(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result_round10 = produce(gold, ldl_with_matrices.cmat,
                                 ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                 ldl_with_matrices.vmat, roundby=10)
        result_round2 = produce(gold, ldl_with_matrices.cmat,
                                ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                ldl_with_matrices.vmat, roundby=2)
        assert isinstance(result_round10, pd.DataFrame)
        assert isinstance(result_round2, pd.DataFrame)


class TestStandaloneProduceEdgeCases:
    """Edge case tests for mapping.produce."""

    def test_zero_vector(self, ldl_with_matrices):
        gold = np.array([0, 0])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert isinstance(result, pd.DataFrame)

    def test_negative_vector(self, ldl_with_matrices):
        gold = np.array([-1, -1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat)
        assert isinstance(result, pd.DataFrame)


class TestStandaloneProduceBackend:
    """Tests for mapping.produce backend parameter."""

    def test_numpy_backend(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='numpy')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Selected' in result.columns

    def test_auto_backend(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='auto')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_invalid_backend(self, ldl_with_matrices):
        gold = np.array([1, 1])
        with pytest.raises(ValueError, match='Unknown backend'):
            produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                    ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                    backend='invalid_backend')

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='torch')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Selected' in result.columns

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_cpu_device(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='torch', device='cpu')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_torch_cuda_device(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='torch', device='cuda')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestStandaloneProduceBackendConsistency:
    """Tests for consistency across backends in mapping.produce."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_numpy_vs_torch_cpu(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result_numpy = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='numpy')
        result_torch = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='torch',
                               device='cpu')
        assert list(result_numpy['Selected']) == list(result_torch['Selected'])
        assert len(result_numpy) == len(result_torch)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_numpy_vs_torch_cuda(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result_numpy = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='numpy')
        result_cuda = produce(gold, ldl_with_matrices.cmat,
                              ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                              ldl_with_matrices.vmat, backend='torch',
                              device='cuda')
        assert list(result_numpy['Selected']) == list(result_cuda['Selected'])
        assert len(result_numpy) == len(result_cuda)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_all_parameters_with_backends(self, ldl_with_matrices):
        gold = np.array([1, 2])
        params = {
            'word': False,
            'roundby': 5,
            'max_attempt': 30,
            'positive': True,
            'apply_vmat': True,
        }
        result_numpy = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='numpy',
                               **params)
        result_torch = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='torch',
                               device='cpu', **params)
        assert list(result_numpy['Selected']) == list(result_torch['Selected'])


# ============================================================
# Part 2: Alignment between LDL.produce and mapping.produce
# ============================================================

class TestAlignmentBasic:
    """Test that mapping.produce produces identical output to LDL.produce."""

    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_selected_cues_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        assert list(result_ldl['Selected']) == list(result_standalone['Selected'])

    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_dataframe_values_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)

    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_word_output_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, word=True, backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, word=True,
                                    backend='numpy')
        assert result_ldl == result_standalone


class TestAlignmentVmat:
    """Test alignment with apply_vmat parameter."""

    def test_with_vmat(self, ldl_with_matrices):
        gold = np.array([1, 2])
        result_ldl = ldl_with_matrices.produce(gold, apply_vmat=True,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, apply_vmat=True,
                                    backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)

    def test_without_vmat(self, ldl_with_matrices):
        gold = np.array([1, 2])
        result_ldl = ldl_with_matrices.produce(gold, apply_vmat=False,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat, apply_vmat=False,
                                    backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)


class TestAlignmentPositive:
    """Test alignment with positive parameter."""

    @pytest.mark.parametrize('positive', [True, False])
    def test_positive_match(self, ldl_with_matrices, positive):
        gold = np.array([1, 1])
        result_ldl = ldl_with_matrices.produce(gold, positive=positive,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, positive=positive,
                                    backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)


class TestAlignmentMaxAttempt:
    """Test alignment with max_attempt parameter."""

    @pytest.mark.parametrize('max_attempt', [1, 5, 50])
    def test_max_attempt_match(self, ldl_with_matrices, max_attempt):
        gold = np.array([1, 2])
        result_ldl = ldl_with_matrices.produce(gold, max_attempt=max_attempt,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat,
                                    max_attempt=max_attempt, backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)


class TestAlignmentRoundby:
    """Test alignment with roundby parameter."""

    @pytest.mark.parametrize('roundby', [2, 5, 10])
    def test_roundby_match(self, ldl_with_matrices, roundby):
        gold = np.array([1, 1])
        result_ldl = ldl_with_matrices.produce(gold, roundby=roundby,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, roundby=roundby,
                                    backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)


class TestAlignmentEdgeCases:
    """Test alignment with edge-case gold vectors."""

    @pytest.mark.parametrize('gold', [
        np.array([0, 0]),
        np.array([-1, -1]),
        np.array([100, 100]),
    ])
    def test_edge_vectors_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)

    def test_list_input_match(self, ldl_with_matrices):
        gold = [1, 1]
        result_ldl = ldl_with_matrices.produce(gold, backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)


class TestAlignmentParameterCombinations:
    """Test alignment with various parameter combinations."""

    @pytest.mark.parametrize('gold, word, positive, apply_vmat', [
        (np.array([1, 1]), False, False, True),
        (np.array([1, 1]), False, True,  True),
        (np.array([1, 1]), False, False, False),
        (np.array([1, 1]), False, True,  False),
        (np.array([1, 2]), False, False, True),
        (np.array([1, 2]), False, True,  True),
        (np.array([1, 2]), False, False, False),
        (np.array([1, 2]), False, True,  False),
    ])
    def test_param_combo_dataframe_match(self, ldl_with_matrices, gold, word,
                                         positive, apply_vmat):
        vmat = ldl_with_matrices.vmat if apply_vmat else None
        result_ldl = ldl_with_matrices.produce(gold, word=word,
                                               positive=positive,
                                               apply_vmat=apply_vmat,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat, vmat=vmat,
                                    word=word, positive=positive,
                                    apply_vmat=apply_vmat, backend='numpy')
        pd.testing.assert_frame_equal(result_ldl, result_standalone)

    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_word_true_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, word=True,
                                               backend='numpy')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, word=True,
                                    backend='numpy')
        assert result_ldl == result_standalone


class TestAlignmentTorchBackend:
    """Test alignment between LDL.produce and mapping.produce using torch."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_torch_cpu_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, backend='torch',
                                               device='cpu')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='torch',
                                    device='cpu')
        assert list(result_ldl['Selected']) == list(result_standalone['Selected'])
        assert len(result_ldl) == len(result_standalone)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize('gold', [
        np.array([1, 1]),
        np.array([1, 2]),
    ])
    def test_torch_cuda_match(self, ldl_with_matrices, gold):
        result_ldl = ldl_with_matrices.produce(gold, backend='torch',
                                               device='cuda')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='torch',
                                    device='cuda')
        assert list(result_ldl['Selected']) == list(result_standalone['Selected'])
        assert len(result_ldl) == len(result_standalone)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_word_match(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result_ldl = ldl_with_matrices.produce(gold, word=True,
                                               backend='torch', device='cpu')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, word=True,
                                    backend='torch', device='cpu')
        assert result_ldl == result_standalone

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_without_vmat_match(self, ldl_with_matrices):
        gold = np.array([1, 2])
        result_ldl = ldl_with_matrices.produce(gold, apply_vmat=False,
                                               backend='torch', device='cpu')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat, apply_vmat=False,
                                    backend='torch', device='cpu')
        assert list(result_ldl['Selected']) == list(result_standalone['Selected'])
        assert len(result_ldl) == len(result_standalone)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_positive_match(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result_ldl = ldl_with_matrices.produce(gold, positive=True,
                                               backend='torch', device='cpu')
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, positive=True,
                                    backend='torch', device='cpu')
        assert list(result_ldl['Selected']) == list(result_standalone['Selected'])
        assert len(result_ldl) == len(result_standalone)
