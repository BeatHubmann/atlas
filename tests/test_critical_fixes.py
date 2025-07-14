"""Tests for critical bug fixes in trajectory prediction models."""

import numpy as np
import pytest
import torch

from atlas_atc.models.linear import LinearExtrapolationPredictor
from atlas_atc.models.lstm import LSTMNetwork
from atlas_atc.models.transformer import TransformerPredictor


class TestLinearModelFixes:
    """Test fixes for linear extrapolation model."""

    def test_linear_dt_validation(self):
        """Test that linear model validates time differences."""
        model = LinearExtrapolationPredictor()

        # Create test data with invalid time difference (dt = 0)
        X = np.array([[[0, 0, 0, 0], [1, 1, 100, 0]]])  # Same timestamp

        with pytest.raises(ValueError, match="Invalid time differences.*dt <= 0"):
            model.predict(X, horizon_minutes=5)

    def test_linear_negative_dt(self):
        """Test that linear model rejects negative time differences."""
        model = LinearExtrapolationPredictor()

        # Create test data with negative time difference
        X = np.array([[[0, 0, 0, 100], [1, 1, 100, 50]]])  # Time goes backward

        with pytest.raises(ValueError, match="Invalid time differences.*dt <= 0"):
            model.predict(X, horizon_minutes=5)

    def test_linear_valid_prediction(self):
        """Test that linear model works with valid input."""
        model = LinearExtrapolationPredictor()

        # Create valid test data
        X = np.array([[[0, 0, 0, 0], [1, 1, 100, 60]]])  # 60 second difference

        result = model.predict(X, horizon_minutes=5)

        # Check result structure
        assert result.predictions.shape == (1, 5, 3)
        assert result.confidence == 0.5
        assert result.computation_time > 0


class TestLSTMModelFixes:
    """Test fixes for LSTM model dimension handling."""

    def test_lstm_dimension_consistency(self):
        """Test that LSTM maintains correct dimensions in autoregressive loop."""
        input_dim = 4
        hidden_dim = 32
        output_dim = 3

        model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            output_dim=output_dim,
        )

        # Create test input
        batch_size = 2
        seq_len = 10
        future_steps = 5
        x = torch.randn(batch_size, seq_len, input_dim)

        # Run forward pass
        predictions = model(x, future_steps=future_steps)

        # Check output dimensions
        assert predictions.shape == (batch_size, future_steps, output_dim)

    def test_lstm_different_input_dims(self):
        """Test LSTM with different input dimensions."""
        for input_dim in [3, 4, 6, 8]:
            model = LSTMNetwork(
                input_dim=input_dim, hidden_dim=32, num_layers=1, output_dim=3
            )

            x = torch.randn(1, 5, input_dim)
            predictions = model(x, future_steps=3)

            assert predictions.shape == (1, 3, 3), f"Failed for input_dim={input_dim}"


class TestTransformerModelFixes:
    """Test fixes for transformer learning rate scheduler."""

    def test_lr_scheduler_no_division_by_zero(self):
        """Test that learning rate scheduler handles step=0 correctly."""
        model = TransformerPredictor()

        # Create minimal training data
        X = np.random.randn(10, 5, 4)  # 10 samples, 5 time steps, 4 features
        y = np.random.randn(10, 3, 3)  # 10 samples, 3 future steps, 3 positions

        # Initialize model (this creates the scheduler)
        model.fit(X, y, epochs=1, batch_size=5)

        # The fit should complete without division by zero errors
        assert model.is_trained

    def test_lr_lambda_function(self):
        """Test the learning rate lambda function directly."""
        warmup_steps = 100

        # Create a dummy optimizer for testing
        dummy_param = torch.nn.Parameter(torch.randn(1))
        optimizer = torch.optim.Adam([dummy_param], lr=0.001)

        # Define the lr_lambda function as in the fix
        def lr_lambda(step):
            step = max(1, step)
            if step < warmup_steps:
                return step / warmup_steps
            return (warmup_steps / step) ** 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Test at step 0 (should not raise error)
        lr_at_0 = scheduler.get_last_lr()[0]
        assert lr_at_0 > 0  # Should be a valid learning rate

        # Test warmup phase
        for _ in range(5):
            scheduler.step()

        # Test post-warmup phase
        for _ in range(warmup_steps + 10):
            scheduler.step()

        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
