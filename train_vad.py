from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from app.models.vad_model import VoiceActivityDetector
from app.services.vad_service import load_wav_to_tensor
from test_vad_service import create_silence_wav


def normalize_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Z-Score стандартизация аудиофрагмента (стандарт в Audio ML)."""
    std = torch.std(chunk)
    if std < 1e-4:
        return torch.zeros_like(chunk)
    mean = torch.mean(chunk)
    return (chunk - mean) / (std + 1e-6)


def train_vad_model():
    print("=" * 50)
    print("ОБУЧЕНИЕ VAD-НЕЙРОСЕТИ СО СТАНДАРТИЗАЦИЕЙ (Z-SCORE)")
    print("=" * 50)

    speech_path = Path("storage/temp/extracted_russian_source_video.wav")
    if not speech_path.exists():
        speech_path = Path("storage/extracted_audio/sample_test.wav")
    
    silence_path = Path("storage/temp/train_silence.wav")
    create_silence_wav(silence_path, duration_sec=5.0)

    speech_tensor = load_wav_to_tensor(speech_path)
    silence_tensor = load_wav_to_tensor(silence_path)

    chunk_size = 8000
    
    # Нарезаем и стандартизируем кусочки
    X_speech = [
        normalize_chunk(speech_tensor[:, :, i:i+chunk_size]) 
        for i in range(0, speech_tensor.shape[2] - chunk_size, chunk_size)
    ]
    X_silence = [
        normalize_chunk(silence_tensor[:, :, i:i+chunk_size]) 
        for i in range(0, silence_tensor.shape[2] - chunk_size, chunk_size)
    ]

    X = torch.cat(X_speech + X_silence, dim=0)
    Y = torch.cat([torch.ones(len(X_speech), 1), torch.zeros(len(X_silence), 1)], dim=0)

    model = VoiceActivityDetector()
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 41):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Эпоха [{epoch}/40] -> Loss: {loss.item():.4f}")

    weights_path = Path("models/vad_weights.pth")
    torch.save(model.state_dict(), str(weights_path))
    print(f"\nМодель переобучена! Веса сохранены: {weights_path}")


if __name__ == "__main__":
    train_vad_model()