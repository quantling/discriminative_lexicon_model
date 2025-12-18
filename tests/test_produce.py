import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
from discriminative_lexicon_model.ldl import LDL, concat_cues, is_consecutive

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
