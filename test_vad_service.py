from pathlib import Path
import wave
import numpy as np

from app.services.vad_service import analyze_audio_speech


def create_silence_wav(output_path: Path, duration_sec: float = 3.0):
    """Генерирует чистый файл тишины (нули в PCM) для проверки негативного кейса."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    
    silence_data = np.zeros(num_samples, dtype=np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Моно
        wav_file.setsampwidth(2)  # 16-бит
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_data.tobytes())


def run_test():
    print("=== ТЕСТ 1: Анализ файла с реальной речью ===")
    speech_audio = Path("storage/temp/extracted_russian_source_video.wav")
    
    if not speech_audio.exists():
        speech_audio = Path("storage/extracted_audio/sample_test.wav")

    speech_result = analyze_audio_speech(speech_audio)
    print(f"Файл: {speech_audio.name}")
    print(f"  Есть речь (has_speech):        {speech_result['has_speech']}")
    print(f"  Доля речи (speech_ratio):      {speech_result['speech_ratio'] * 100}%")
    print(f"  Окон проанализировано:        {speech_result['total_windows_analyzed']}")
    print(f"  Найдено речевых фрагментов:    {speech_result['speech_windows_found']}")
    print("  Интервалы речи:")
    for interval in speech_result["speech_intervals"][:3]:
        print(f"    [{interval['start']}s -> {interval['end']}s] Уверенность: {interval['confidence']}")

    print("\n=== ТЕСТ 2: Анализ файла с тишиной (Silence Test) ===")
    silence_audio = Path("storage/temp/pure_silence.wav")
    create_silence_wav(silence_audio, duration_sec=3.0)

    silence_result = analyze_audio_speech(silence_audio)
    print(f"Файл: {silence_audio.name}")
    print(f"  Есть речь (has_speech):        {silence_result['has_speech']}")
    print(f"  Доля речи (speech_ratio):      {silence_result['speech_ratio'] * 100}%")
    print(f"  Найдено речевых фрагментов:    {silence_result['speech_windows_found']}")

    assert speech_result["has_speech"] is True, "Ошибка: Речь не была обнаружена!"
    assert silence_result["has_speech"] is False, "Ошибка: В тишине ложно обнаружена речь!"
    print("\n" + "=" * 50)
    print("УСПЕХ: VAD-НЕЙРОСЕТЬ ИДЕАЛЬНО РАЗЛИЧАЕТ ГОЛОС И ТИШИНУ!")
    print("=" * 50)


if __name__ == "__main__":
    run_test()