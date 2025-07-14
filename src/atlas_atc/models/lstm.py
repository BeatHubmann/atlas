"""LSTM trajectory prediction model."""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional
from .base import TorchTrajectoryPredictor, PredictionResult


class LSTMNetwork(nn.Module):
    """LSTM network for trajectory prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            future_steps: Number of future steps to predict
        
        Returns:
            Predictions [batch_size, future_steps, output_dim]
        """
        batch_size = x.size(0)
        
        # Process input sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Generate future predictions
        predictions = []
        hidden = (h_n, c_n)
        
        # Use last input as starting point for autoregressive prediction
        current_input = x[:, -1:, :]  # [batch_size, 1, input_dim]
        
        for _ in range(future_steps):
            # Predict next step
            lstm_out, hidden = self.lstm(current_input, hidden)
            output = self.output_proj(self.dropout(lstm_out))
            predictions.append(output)
            
            # Update input for next prediction (use predicted position + time)
            # For simplicity, we'll use the prediction as next input
            # In practice, you'd need to handle time features properly
            current_input = output
        
        return torch.cat(predictions, dim=1)


class LSTMPredictor(TorchTrajectoryPredictor):
    """LSTM-based trajectory predictor."""
    
    def __init__(self, device: str = "mps", hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__("LSTM", device)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Model will be initialized in fit()
        self.model: Optional[LSTMNetwork] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            learning_rate: float = 0.001, batch_size: int = 32) -> None:
        """
        Train LSTM model.
        
        Args:
            X: Input sequences [num_samples, seq_len, features]
            y: Target sequences [num_samples, target_len, 3]
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        input_dim = X.shape[2]
        output_dim = y.shape[2]
        
        # Initialize model
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Convert to tensors
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X, future_steps=batch_y.size(1))
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray, horizon_minutes: int) -> PredictionResult:
        """
        Predict future trajectory using trained LSTM.
        
        Args:
            X: Historical trajectory points [batch_size, seq_len, features]
            horizon_minutes: Prediction horizon in minutes
        
        Returns:
            PredictionResult with predicted positions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            
            # Generate predictions
            predictions = self.model(X_tensor, future_steps=horizon_minutes)
            predictions_np = self._to_numpy(predictions)
        
        computation_time = time.time() - start_time
        
        return PredictionResult(
            predictions=predictions_np,
            confidence=0.85,  # Higher confidence for trained model
            computation_time=computation_time,
            metadata={
                "model": self.name,
                "horizon_minutes": horizon_minutes,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "parameters": sum(p.numel() for p in self.model.parameters())
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        
        return {
            "name": self.name,
            "type": "neural_network",
            "parameters": params,
            "training_required": True,
            "architecture": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout
            },
            "description": "LSTM network with autoregressive prediction"
        }

