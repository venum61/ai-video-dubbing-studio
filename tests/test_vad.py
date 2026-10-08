import torch
import numpy as np
from app.models.vad_model import VoiceActivityDetector
from app.services.vad_service import analyze_audio_speech


def test_vad_model_forward():
    """Тест 1: Проверка размерностей и выхода собственной PyTorch модели."""
    model = VoiceActivityDetector()
    model.eval()

    dummy_input = torch.randn(2, 1, 16000)
    
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), f"Ожидалась форма (2, 1), получена {output.shape}"
    
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Вероятности вышли за пределы [0, 1]"


def test_vad_model_dynamic_length():
    """Тест 2: Проверка работы адаптивного пулинга на разной длине звука."""
    model = VoiceActivityDetector()
    model.eval()

    for length in [4000, 8000, 32000]:
        test_chunk = torch.randn(1, 1, length)
        with torch.no_grad():
            res = model(test_chunk)
        assert res.shape == (1, 1)
        assert not torch.isnan(res).any(), "Выход содержит NaN!"