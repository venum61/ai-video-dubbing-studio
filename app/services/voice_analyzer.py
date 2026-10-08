from pathlib import Path
from typing import Union, Dict, Any, List
import wave
import numpy as np
import logging

logger = logging.getLogger(__name__)


def estimate_pitch_f0(audio_mono: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Вычисляет фундаментальную частоту основного тона (F0 / Pitch в Гц)
    с помощью метода автокорреляции звукового сигнала.
    """
    
    if len(audio_mono) < sample_rate * 0.2:
        return 130.0

    signal = audio_mono - np.mean(audio_mono)
    
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]

    min_period = int(sample_rate / 350)  # ~45 отсчетов
    max_period = int(sample_rate / 70)   # ~228 отсчетов

    if len(corr) < max_period:
        return 130.0

    peak_index = np.argmax(corr[min_period:max_period]) + min_period
    pitch_hz = sample_rate / peak_index

    return round(float(pitch_hz), 1)


def analyze_speaker_voice(
    audio_path: Union[str, Path], 
    speech_segments: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Анализирует оригинальный голос спикера из видео:
    определяет пол, высоту тона (Pitch) и подбирает тип нейро-актера дубляжа.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    # Читаем WAV файл
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)

    audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    if speech_segments and len(speech_segments) > 0:
        first_seg = speech_segments[0]
        start_idx = int(first_seg.get("start", 0) * sample_rate)
        end_idx = int(first_seg.get("end", start_idx + sample_rate * 3) * sample_rate)
        sample_audio = audio_np[start_idx:end_idx]
    else:
        sample_audio = audio_np[:sample_rate * 5]  # Первые 5 секунд

    if len(sample_audio) == 0:
        sample_audio = audio_np

    pitch_hz = estimate_pitch_f0(sample_audio, sample_rate)

    if pitch_hz < 165.0:
        gender = "male"
        voice_desc = "Мужской тембр (Бас/Баритон)"
    else:
        gender = "female"
        voice_desc = "Женский / Высокий тембр"

    logger.info(f"Анализ голоса спикера: {voice_desc} (F0 = {pitch_hz} Гц)")

    return {
        "gender": gender,
        "pitch_hz": pitch_hz,
        "voice_description": voice_desc
    }