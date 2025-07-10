# pylint: disable=C0114,C0115,C0116
import torch
from torch import nn


class AttentionAggregator(nn.Module):
    """Aggregates input features using attention mechanism."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.attention_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_dim)  # Flatten: [N, C * H * W]
        features = self.feature_transform(x)  # [N, output_dim]
        attention_scores = self.attention_net(features)  # [N, 1]
        attention_weights = torch.softmax(attention_scores.T, dim=1)  # [1, N]
        aggregated = torch.mm(attention_weights, features)  # [1, output_dim]
        return aggregated, attention_weights


class MILClassifier(nn.Module):
    """Multiple instance learning classifier with attention-based aggregation."""

    def __init__(self, num_classes: int, input_dim: int = 1024):
        super().__init__()

        self.aggregator = AttentionAggregator(input_dim=input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.aggregator.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor = None):
        # inputs: [1, N, C, H, W] → [N, C, H, W]
        inputs = inputs.squeeze(0)
        aggregated_features, attention_weights = self.aggregator(inputs)
        logits = self.classifier(aggregated_features)
        return logits, attention_weights
