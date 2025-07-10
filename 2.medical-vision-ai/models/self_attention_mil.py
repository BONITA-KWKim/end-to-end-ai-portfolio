# pylint: disable=C0114,C0115,C0116
import torch
from torch import nn
from .nystrom_attention import NystromAttention


class SelfAttentionAggregator(nn.Module):
    """Aggregates input features using Nystrom-based self-attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.attention_layer = NystromAttention(output_dim)

        self.score_layer = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Dropout(),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, N, F]
        x = self.feature_transform(x)  # [B, N, output_dim]
        attention_features = self.attention_layer(x)  # [B, N, output_dim]
        attention_scores = self.score_layer(attention_features)  # [B, N, 1]

        attention_scores = attention_scores.squeeze(0).T  # [1, N]
        attention_weights = torch.softmax(attention_scores, dim=1)

        x = x.squeeze(0)  # [N, output_dim]
        aggregated = torch.mm(attention_weights, x)  # [1, output_dim]

        return aggregated, attention_weights


class MILSelfAttentionClassifier(nn.Module):
    """MIL classifier with Nystrom self-attention-based aggregation."""

    def __init__(self, num_classes: int, input_dim: int = 1024):
        super().__init__()

        self.aggregator = SelfAttentionAggregator(input_dim=input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.aggregator.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor, aux: torch.Tensor = None):
        # inputs: [1, N, F] – no need to squeeze if already batched
        aggregated_features, attention_weights = self.aggregator(inputs)
        logits = self.classifier(aggregated_features)
        return logits, attention_weights
