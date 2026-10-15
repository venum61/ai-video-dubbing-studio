from pathlib import Path
from typing import Union, Dict, Any, List
import wave
import numpy as np
import torch
import torch.nn.functional as F
import logging

from app.models.vad_model import VoiceActivityDetector

logger = logging.getLogger(__name__)

_VAD_MODEL = VoiceActivityDetector()
WEIGHTS_PATH = Path("models/vad_weights.pth")

if WEIGHTS_PATH.exists():
    _VAD_MODEL.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu"))
    logger.info(f"Загружены веса VAD: {WEIGHTS_PATH}")

_VAD_MODEL.eval()


def load_wav_to_tensor(audio_path: Union[str, Path]) -> torch.Tensor:
    """Считывает WAV-файл в тензор PyTorch."""
    with wave.open(str(audio_path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)

    audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    if n_channels > 1:
        audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

    audio_np = audio_np / 32768.0
    return torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)


def analyze_audio_speech(
    audio_path: Union[str, Path], 
    threshold: float = 0.5,
    window_sec: float = 0.5
) -> Dict[str, Any]:
    """
    Супер-быстрый батчевый инференс: прогоняет все окна аудио за 1 матричный вызов PyTorch.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    audio_tensor = load_wav_to_tensor(audio_path)
    total_samples = audio_tensor.shape[2]
    sample_rate = 16000
    window_samples = int(sample_rate * window_sec)

    chunks = []
    metadata = []

    # 1. Быстро нарезаем и нормализуем все окна в памяти
    for start_idx in range(0, total_samples, window_samples):
        end_idx = min(start_idx + window_samples, total_samples)
        chunk = audio_tensor[:, :, start_idx:end_idx]

        if chunk.shape[2] < window_samples:
            pad_size = window_samples - chunk.shape[2]
            chunk = F.pad(chunk, (0, pad_size), mode="constant", value=0.0)

        std = torch.std(chunk)
        if std < 1e-4:
            norm_chunk = torch.zeros_like(chunk)
            is_pure_silence = True
        else:
            norm_chunk = (chunk - torch.mean(chunk)) / (std + 1e-6)
            is_pure_silence = False

        chunks.append(norm_chunk)
        metadata.append({
            "start": round(start_idx / sample_rate, 2),
            "end": round(end_idx / sample_rate, 2),
            "is_pure_silence": is_pure_silence
        })

    if not chunks:
        return {"has_speech": False, "speech_ratio": 0.0, "speech_intervals": []}

    # 2. Объединяем все окна в один большой батч [Batch_Size, 1, 8000]
    batch_tensor = torch.cat(chunks, dim=0)

    # 3. ЕДИНЫЙ МАТРИЧНЫЙ ПРОХОД НЕЙРОСЕТИ
    with torch.no_grad():
        probabilities = _VAD_MODEL(batch_tensor).squeeze(-1).tolist()

    if isinstance(probabilities, float):
        probabilities = [probabilities]

    # 4. Сбор интервалов
    speech_intervals = []
    speech_windows_count = 0

    for prob, meta in zip(probabilities, metadata):
        # Если тишина (std = 0), зануляем вероятность
        final_prob = 0.0 if meta["is_pure_silence"] else prob

        if final_prob >= threshold:
            speech_windows_count += 1
            speech_intervals.append({
                "start": meta["start"],
                "end": meta["end"],
                "confidence": round(final_prob, 4)
            })

    total_windows = len(probabilities)
    speech_ratio = round(speech_windows_count / total_windows, 2) if total_windows > 0 else 0.0

    return {
        "has_speech": speech_windows_count > 0,
        "speech_ratio": speech_ratio,
        "total_windows_analyzed": total_windows,
        "speech_windows_found": speech_windows_count,
        "speech_intervals": speech_intervals
    }