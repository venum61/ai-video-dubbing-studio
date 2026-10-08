import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector(nn.Module):
    """
    Собственная 1D-сверточная нейронная сеть на PyTorch 
    для бинарной классификации аудиофрагментов (Речь vs Тишина/Шум).
    """
    def __init__(self, in_channels: int = 1):
        super(VoiceActivityDetector, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=16, 
            kernel_size=64, 
            stride=16
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=32, 
            stride=8
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=16, 
            stride=4
        )
        self.bn3 = nn.BatchNorm1d(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход нейросети: Аудио -> Свертки -> Признаки -> Вероятность речи.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        logits = self.fc(x)
        probabilities = torch.sigmoid(logits)

        return probabilities