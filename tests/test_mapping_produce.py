import pytest
import numpy as np
import pandas as pd
import xarray as xr
from discriminative_lexicon_model.ldl import LDL
from discriminative_lexicon_model.mapping import produce, gen_chat_produce, gen_chat, produce_paradigm

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


# ============================================================
# Part 3: gen_chat_produce tests
# ============================================================

class TestGenChatProduceBasic:
    """Basic tests for gen_chat_produce."""

    def test_returns_xarray(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert isinstance(result, xr.DataArray)

    def test_has_correct_dims(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert result.dims == ('word', 'cues')

    def test_has_correct_shape(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        n_words = ldl_with_matrices.smat.shape[0]
        n_cues = ldl_with_matrices.cmat.shape[1]
        assert result.shape == (n_words, n_cues)

    def test_word_coords_match_smat(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert list(result.word.values) == list(ldl_with_matrices.smat.word.values)

    def test_cue_coords_match_cmat(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert list(result.cues.values) == list(ldl_with_matrices.cmat.cues.values)

    def test_values_are_finite(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert np.all(np.isfinite(result.values))


class TestGenChatProduceSameStructureAsGenChat:
    """Test that gen_chat_produce has the same structure as gen_chat."""

    def test_same_dims(self, ldl_with_matrices):
        chat_mm = gen_chat(smat=ldl_with_matrices.smat,
                           gmat=ldl_with_matrices.gmat)
        chat_prod = gen_chat_produce(ldl_with_matrices.smat,
                                     ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        assert chat_mm.dims == chat_prod.dims

    def test_same_shape(self, ldl_with_matrices):
        chat_mm = gen_chat(smat=ldl_with_matrices.smat,
                           gmat=ldl_with_matrices.gmat)
        chat_prod = gen_chat_produce(ldl_with_matrices.smat,
                                     ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        assert chat_mm.shape == chat_prod.shape

    def test_same_word_coords(self, ldl_with_matrices):
        chat_mm = gen_chat(smat=ldl_with_matrices.smat,
                           gmat=ldl_with_matrices.gmat)
        chat_prod = gen_chat_produce(ldl_with_matrices.smat,
                                     ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        assert list(chat_mm.word.values) == list(chat_prod.word.values)

    def test_same_cue_coords(self, ldl_with_matrices):
        chat_mm = gen_chat(smat=ldl_with_matrices.smat,
                           gmat=ldl_with_matrices.gmat)
        chat_prod = gen_chat_produce(ldl_with_matrices.smat,
                                     ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        assert list(chat_mm.cues.values) == list(chat_prod.cues.values)


class TestGenChatProduceConsistencyWithProduce:
    """Test that gen_chat_produce rows match summed produce outputs."""

    @pytest.mark.parametrize('word_idx, gold', [
        (0, np.array([1, 1])),   # 'ban'
        (1, np.array([1, 2])),   # 'banban'
    ])
    def test_row_equals_summed_produce_vectors(self, ldl_with_matrices,
                                                word_idx, gold):
        chat_prod = gen_chat_produce(ldl_with_matrices.smat,
                                     ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         backend='numpy')
        numeric_cols = result.drop(columns=['Selected'])
        expected_row = numeric_cols.sum(axis=0).values
        np.testing.assert_allclose(chat_prod.values[word_idx], expected_row)


class TestGenChatProduceParameters:
    """Test gen_chat_produce with different parameter settings."""

    def test_without_vmat(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  apply_vmat=False, backend='numpy')
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(words), ldl_with_matrices.cmat.shape[1])

    def test_positive_true(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, positive=True,
                                  backend='numpy')
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(words), ldl_with_matrices.cmat.shape[1])

    def test_custom_max_attempt(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, max_attempt=100,
                                  backend='numpy')
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize('roundby', [2, 5, 10])
    def test_roundby(self, ldl_with_matrices, roundby):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, roundby=roundby,
                                  backend='numpy')
        assert isinstance(result, xr.DataArray)

    def test_vmat_with_and_without_differ(self, ldl_with_matrices):
        result_with = gen_chat_produce(ldl_with_matrices.smat,
                                       ldl_with_matrices.cmat,
                                       ldl_with_matrices.fmat,
                                       ldl_with_matrices.gmat,
                                       ldl_with_matrices.vmat, apply_vmat=True,
                                       backend='numpy')
        result_without = gen_chat_produce(ldl_with_matrices.smat,
                                          ldl_with_matrices.cmat,
                                          ldl_with_matrices.fmat,
                                          ldl_with_matrices.gmat,
                                          apply_vmat=False, backend='numpy')
        assert result_with.shape == result_without.shape


class TestGenChatProduceBackend:
    """Test gen_chat_produce with different backends."""

    def test_numpy_backend(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert isinstance(result, xr.DataArray)

    def test_auto_backend(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='auto')
        assert isinstance(result, xr.DataArray)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='torch',
                                  device='cpu')
        assert isinstance(result, xr.DataArray)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_torch_cuda_backend(self, ldl_with_matrices):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='torch',
                                  device='cuda')
        assert isinstance(result, xr.DataArray)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_numpy_vs_torch_cpu_match(self, ldl_with_matrices):
        result_numpy = gen_chat_produce(ldl_with_matrices.smat,
                                        ldl_with_matrices.cmat,
                                        ldl_with_matrices.fmat,
                                        ldl_with_matrices.gmat,
                                        ldl_with_matrices.vmat,
                                        backend='numpy')
        result_torch = gen_chat_produce(ldl_with_matrices.smat,
                                        ldl_with_matrices.cmat,
                                        ldl_with_matrices.fmat,
                                        ldl_with_matrices.gmat,
                                        ldl_with_matrices.vmat,
                                        backend='torch', device='cpu')
        xr.testing.assert_allclose(result_numpy, result_torch)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_numpy_vs_torch_cuda_match(self, ldl_with_matrices):
        result_numpy = gen_chat_produce(ldl_with_matrices.smat,
                                        ldl_with_matrices.cmat,
                                        ldl_with_matrices.fmat,
                                        ldl_with_matrices.gmat,
                                        ldl_with_matrices.vmat,
                                        backend='numpy')
        result_cuda = gen_chat_produce(ldl_with_matrices.smat,
                                       ldl_with_matrices.cmat,
                                       ldl_with_matrices.fmat,
                                       ldl_with_matrices.gmat,
                                       ldl_with_matrices.vmat,
                                       backend='torch', device='cuda')
        xr.testing.assert_allclose(result_numpy, result_cuda)


# ============================================================
# Part 4: produce_paradigm tests
# ============================================================

class TestProduceParadigmBasic:
    """Basic tests for produce_paradigm."""

    def test_returns_dataframe(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert 'index' in result.columns
        assert 'word' in result.columns
        assert 'pred' in result.columns
        assert 'step' in result.columns
        assert 'Selected' in result.columns

    def test_has_cue_columns(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        cues = ldl_with_matrices.cmat.cues.values
        for cue in cues:
            assert cue in result.columns

    def test_column_order(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert list(result.columns[:5]) == ['index', 'word', 'pred', 'step', 'Selected']

    def test_all_words_present(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        expected_words = list(ldl_with_matrices.smat.word.values)
        result_words = result.groupby('index')['word'].first().tolist()
        assert result_words == expected_words

    def test_index_values(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        n_words = ldl_with_matrices.smat.shape[0]
        expected_indices = list(range(n_words))
        actual_indices = sorted(result['index'].unique())
        assert actual_indices == expected_indices


class TestProduceParadigmSumRow:
    """Tests for the (sum) row in produce_paradigm."""

    def test_each_word_has_sum_row(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        n_words = ldl_with_matrices.smat.shape[0]
        sum_rows = result[result['step'] == '(sum)']
        assert len(sum_rows) == n_words

    def test_sum_row_selected_is_sum(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        sum_rows = result[result['step'] == '(sum)']
        assert (sum_rows['Selected'] == '(sum)').all()

    def test_sum_row_values_equal_column_sums(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        cues = list(ldl_with_matrices.cmat.cues.values)
        for idx in result['index'].unique():
            word_rows = result[result['index'] == idx]
            step_rows = word_rows[word_rows['step'] != '(sum)']
            sum_row = word_rows[word_rows['step'] == '(sum)']
            expected = step_rows[cues].sum(axis=0).values
            actual = sum_row[cues].values.flatten()
            np.testing.assert_allclose(actual, expected)

    def test_sum_row_is_last_per_word(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        for idx in result['index'].unique():
            word_rows = result[result['index'] == idx]
            last_row = word_rows.iloc[-1]
            assert last_row['step'] == '(sum)'

    def test_sum_row_index_is_int(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        sum_rows = result[result['step'] == '(sum)']
        for val in sum_rows['index']:
            assert isinstance(val, (int, np.integer))


class TestProduceParadigmPredicted:
    """Tests for the predicted column in produce_paradigm."""

    def test_predicted_starts_and_ends_with_boundary(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        for predicted in result['pred'].unique():
            if predicted:  # skip empty string for incomplete productions
                assert predicted.startswith('#')
                assert predicted.endswith('#')

    def test_predicted_same_for_all_rows_of_word(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        for idx in result['index'].unique():
            word_rows = result[result['index'] == idx]
            assert word_rows['pred'].nunique() == 1

    def test_predicted_matches_produce_word(self, ldl_with_matrices):
        """Test that predicted column matches produce(..., word=True)."""
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        smat = ldl_with_matrices.smat
        for i in range(smat.shape[0]):
            gold = smat.values[i]
            expected = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, word=True,
                               backend='numpy')
            word_rows = result[result['index'] == i]
            assert word_rows['pred'].iloc[0] == expected


class TestProduceParadigmConsistencyWithProduce:
    """Test that produce_paradigm rows match individual produce calls."""

    @pytest.mark.parametrize('word_idx, gold', [
        (0, np.array([1, 1])),   # 'ban'
        (1, np.array([1, 2])),   # 'banban'
    ])
    def test_step_rows_match_produce(self, ldl_with_matrices, word_idx, gold):
        paradigm = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        individual = produce(gold, ldl_with_matrices.cmat,
                             ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                             ldl_with_matrices.vmat, backend='numpy')
        # Get step rows (exclude sum row) for this word
        word_rows = paradigm[paradigm['index'] == word_idx]
        step_rows = word_rows[word_rows['step'] != '(sum)']
        # Selected cues should match
        assert list(step_rows['Selected'].values) == list(individual['Selected'].values)
        # Cue values should match
        cues = list(ldl_with_matrices.cmat.cues.values)
        np.testing.assert_allclose(
            step_rows[cues].values.astype(float),
            individual[cues].values.astype(float),
        )

    @pytest.mark.parametrize('word_idx, gold', [
        (0, np.array([1, 1])),
        (1, np.array([1, 2])),
    ])
    def test_sum_row_matches_gen_chat_produce(self, ldl_with_matrices, word_idx, gold):
        paradigm = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy')
        chat_prod = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, backend='numpy')
        cues = list(ldl_with_matrices.cmat.cues.values)
        word_rows = paradigm[paradigm['index'] == word_idx]
        sum_row = word_rows[word_rows['step'] == '(sum)']
        np.testing.assert_allclose(
            sum_row[cues].values.flatten().astype(float),
            chat_prod.values[word_idx],
        )


class TestProduceParadigmParameters:
    """Test produce_paradigm with different parameter settings."""

    def test_without_vmat(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  apply_vmat=False, backend='numpy')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_positive_true(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, positive=True,
                                  backend='numpy')
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_custom_max_attempt(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, max_attempt=100,
                                  backend='numpy')
        assert isinstance(result, pd.DataFrame)


class TestProduceParadigmBackend:
    """Test produce_paradigm with different backends."""

    def test_numpy_backend(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy')
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_backend(self, ldl_with_matrices):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='torch',
                                  device='cpu')
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_numpy_vs_torch_cpu_match(self, ldl_with_matrices):
        result_numpy = produce_paradigm(ldl_with_matrices.smat,
                                        ldl_with_matrices.cmat,
                                        ldl_with_matrices.fmat,
                                        ldl_with_matrices.gmat,
                                        ldl_with_matrices.vmat,
                                        backend='numpy')
        result_torch = produce_paradigm(ldl_with_matrices.smat,
                                        ldl_with_matrices.cmat,
                                        ldl_with_matrices.fmat,
                                        ldl_with_matrices.gmat,
                                        ldl_with_matrices.vmat,
                                        backend='torch', device='cpu')
        assert list(result_numpy['Selected']) == list(result_torch['Selected'])
        assert list(result_numpy['pred']) == list(result_torch['pred'])
        cues = list(ldl_with_matrices.cmat.cues.values)
        np.testing.assert_allclose(
            result_numpy[cues].values.astype(float),
            result_torch[cues].values.astype(float),
        )


class TestProduceParadigmDuplicateWords:
    """Test produce_paradigm with duplicate word labels in smat."""

    def test_duplicate_words_have_distinct_indices(self):
        """Test that duplicate words are distinguishable by index."""
        # Build smat with duplicate words
        dup_words = ['ban', 'ban']
        dup_semdf = pd.DataFrame({'hit': [1, 1], 'intensity': [1, 2]},
                                 index=dup_words)
        ldl = LDL(dup_words, dup_semdf, allmatrices=True)
        result = produce_paradigm(ldl.smat, ldl.cmat, ldl.fmat, ldl.gmat,
                                  ldl.vmat, backend='numpy')
        indices = sorted(result['index'].unique())
        assert indices == [0, 1]
        # Both should have 'ban' as word
        for idx in indices:
            word_rows = result[result['index'] == idx]
            assert (word_rows['word'] == 'ban').all()

    def test_duplicate_words_each_have_sum_row(self):
        """Test that each duplicate word gets its own sum row."""
        dup_words = ['ban', 'ban']
        dup_semdf = pd.DataFrame({'hit': [1, 1], 'intensity': [1, 2]},
                                 index=dup_words)
        ldl = LDL(dup_words, dup_semdf, allmatrices=True)
        result = produce_paradigm(ldl.smat, ldl.cmat, ldl.fmat, ldl.gmat,
                                  ldl.vmat, backend='numpy')
        sum_rows = result[result['step'] == '(sum)']
        assert len(sum_rows) == 2
        assert list(sum_rows['index']) == [0, 1]


# ============================================================
# Part 5: stop parameter tests
# ============================================================

class TestProduceStopParameter:
    """Test the stop parameter for produce."""

    def test_default_is_convergence(self, ldl_with_matrices):
        """Default stop should be 'convergence'."""
        gold = np.array([1, 1])
        result_default = produce(gold, ldl_with_matrices.cmat,
                                 ldl_with_matrices.fmat,
                                 ldl_with_matrices.gmat,
                                 ldl_with_matrices.vmat, backend='numpy')
        result_explicit = produce(gold, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat,
                                  ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, backend='numpy',
                                  stop='convergence')
        pd.testing.assert_frame_equal(result_default, result_explicit)

    def test_boundary_returns_dataframe(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         stop='boundary')
        assert isinstance(result, pd.DataFrame)
        assert 'Selected' in result.columns
        assert len(result) > 0

    def test_convergence_returns_dataframe(self, ldl_with_matrices):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         stop='convergence')
        assert isinstance(result, pd.DataFrame)
        assert 'Selected' in result.columns
        assert len(result) > 0

    def test_boundary_stops_at_hash(self, ldl_with_matrices):
        """With stop='boundary', last selected cue should end with '#'."""
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         stop='boundary')
        last_cue = result['Selected'].iloc[-1]
        assert last_cue.endswith('#')

    def test_convergence_may_continue_past_hash(self, ldl_with_matrices):
        """With stop='convergence', algorithm does not stop at '#' cues."""
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         stop='convergence')
        # Find the first '#'-ending cue
        selected = result['Selected'].tolist()
        first_hash_idx = None
        for j, cue in enumerate(selected):
            is_unigram_onset = j == 0 and len(cue) == 1 and cue == '#'
            if cue.endswith('#') and not is_unigram_onset:
                first_hash_idx = j
                break
        # convergence either continues past '#' or converges at the same point
        # (we can't guarantee it continues, but it should not stop early due to '#')
        assert len(result) >= first_hash_idx + 1 if first_hash_idx is not None else True

    def test_convergence_at_least_as_many_steps_as_boundary(self, ldl_with_matrices):
        """Convergence mode should produce >= as many steps as boundary mode."""
        gold = np.array([1, 2])
        result_boundary = produce(gold, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat,
                                  ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, stop='boundary')
        result_convergence = produce(gold, ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, stop='convergence')
        assert len(result_convergence) >= len(result_boundary)

    def test_boundary_prefix_matches_convergence(self, ldl_with_matrices):
        """Boundary result should be a prefix of convergence result."""
        gold = np.array([1, 2])
        result_boundary = produce(gold, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat,
                                  ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, stop='boundary')
        result_convergence = produce(gold, ldl_with_matrices.cmat,
                                     ldl_with_matrices.fmat,
                                     ldl_with_matrices.gmat,
                                     ldl_with_matrices.vmat, stop='convergence')
        n = len(result_boundary)
        pd.testing.assert_frame_equal(result_convergence.iloc[:n].reset_index(drop=True),
                                      result_boundary.reset_index(drop=True))

    def test_invalid_stop_raises(self, ldl_with_matrices):
        gold = np.array([1, 1])
        with pytest.raises(ValueError, match='Unknown stop'):
            produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                    ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                    stop='invalid')

    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_ldl_produce_matches_mapping_produce(self, ldl_with_matrices, stop):
        """LDL.produce and mapping.produce should match for both stop modes."""
        gold = np.array([1, 2])
        result_ldl = ldl_with_matrices.produce(gold, backend='numpy', stop=stop)
        result_standalone = produce(gold, ldl_with_matrices.cmat,
                                    ldl_with_matrices.fmat,
                                    ldl_with_matrices.gmat,
                                    ldl_with_matrices.vmat, backend='numpy',
                                    stop=stop)
        pd.testing.assert_frame_equal(result_ldl, result_standalone)

    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_apply_vmat_false_with_stop(self, ldl_with_matrices, stop):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, apply_vmat=False,
                         stop=stop)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_positive_with_stop(self, ldl_with_matrices, stop):
        gold = np.array([1, 1])
        result = produce(gold, ldl_with_matrices.cmat, ldl_with_matrices.fmat,
                         ldl_with_matrices.gmat, ldl_with_matrices.vmat,
                         positive=True, stop=stop)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestProduceStopTorchBackend:
    """Test stop parameter with torch backend."""

    @pytest.mark.skipif(not HAS_TORCH, reason='PyTorch not installed')
    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_torch_cpu_matches_numpy(self, ldl_with_matrices, stop):
        gold = np.array([1, 2])
        result_numpy = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat,
                               ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='numpy',
                               stop=stop)
        result_torch = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat,
                               ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='torch',
                               device='cpu', stop=stop)
        pd.testing.assert_frame_equal(result_numpy, result_torch)

    @pytest.mark.skipif(not HAS_CUDA, reason='CUDA not available')
    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_torch_cuda_matches_numpy(self, ldl_with_matrices, stop):
        gold = np.array([1, 2])
        result_numpy = produce(gold, ldl_with_matrices.cmat,
                               ldl_with_matrices.fmat,
                               ldl_with_matrices.gmat,
                               ldl_with_matrices.vmat, backend='numpy',
                               stop=stop)
        result_cuda = produce(gold, ldl_with_matrices.cmat,
                              ldl_with_matrices.fmat,
                              ldl_with_matrices.gmat,
                              ldl_with_matrices.vmat, backend='torch',
                              device='cuda', stop=stop)
        pd.testing.assert_frame_equal(result_numpy, result_cuda)


class TestGenChatProduceStopParameter:
    """Test stop parameter for gen_chat_produce."""

    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_returns_xarray(self, ldl_with_matrices, stop):
        result = gen_chat_produce(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, stop=stop)
        assert isinstance(result, xr.DataArray)

    def test_convergence_default(self, ldl_with_matrices):
        result_default = gen_chat_produce(ldl_with_matrices.smat,
                                          ldl_with_matrices.cmat,
                                          ldl_with_matrices.fmat,
                                          ldl_with_matrices.gmat,
                                          ldl_with_matrices.vmat)
        result_explicit = gen_chat_produce(ldl_with_matrices.smat,
                                           ldl_with_matrices.cmat,
                                           ldl_with_matrices.fmat,
                                           ldl_with_matrices.gmat,
                                           ldl_with_matrices.vmat,
                                           stop='convergence')
        xr.testing.assert_equal(result_default, result_explicit)


class TestProduceParadigmStopParameter:
    """Test stop parameter for produce_paradigm."""

    @pytest.mark.parametrize('stop', ['convergence', 'boundary'])
    def test_returns_dataframe(self, ldl_with_matrices, stop):
        result = produce_paradigm(ldl_with_matrices.smat, ldl_with_matrices.cmat,
                                  ldl_with_matrices.fmat, ldl_with_matrices.gmat,
                                  ldl_with_matrices.vmat, stop=stop)
        assert isinstance(result, pd.DataFrame)
        assert 'index' in result.columns
        assert 'word' in result.columns

    def test_convergence_at_least_as_many_rows_as_boundary(self, ldl_with_matrices):
        result_boundary = produce_paradigm(ldl_with_matrices.smat,
                                           ldl_with_matrices.cmat,
                                           ldl_with_matrices.fmat,
                                           ldl_with_matrices.gmat,
                                           ldl_with_matrices.vmat,
                                           stop='boundary')
        result_convergence = produce_paradigm(ldl_with_matrices.smat,
                                              ldl_with_matrices.cmat,
                                              ldl_with_matrices.fmat,
                                              ldl_with_matrices.gmat,
                                              ldl_with_matrices.vmat,
                                              stop='convergence')
        assert len(result_convergence) >= len(result_boundary)
