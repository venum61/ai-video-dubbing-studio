from pathlib import Path
from gtts import gTTS

from app.services.transcriber import transcribe_audio


def run_test():
    print("=== ТЕСТ 1: Генерация тестовой речи ===")
    test_audio = Path("storage/extracted_audio/sample_speech.mp3")
    test_audio.parent.mkdir(parents=True, exist_ok=True)

    phrase = "Привет! Это тестовая запись для проверки системы переозвучки видео."
    print(f"Текст для синтеза: '{phrase}'")

    tts = gTTS(text=phrase, lang="ru")
    tts.save(str(test_audio))
    print(f"Файл создан: {test_audio}")

    print("\n=== ТЕСТ 2: Распознавание речи через Whisper ===")
    result = transcribe_audio(test_audio, model_size="base")

    print("\n" + "=" * 40)
    print("ИТОГИ РАСПОЗНАВАНИЯ:")
    print("=" * 40)
    print(f"Определенный язык: {result['language'].upper()}")
    print(f"Распознанный текст:\n{result['text']}")
    print("\nСегменты с таймкодами:")
    for seg in result["segments"]:
        print(f"  [{seg['start']}s -> {seg['end']}s]: {seg['text']}")
    print("=" * 40)


if __name__ == "__main__":
    run_test()