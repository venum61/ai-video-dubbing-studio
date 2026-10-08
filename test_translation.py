from app.services.translator import translate_text, translate_segments

def run_test():
    print("=== ТЕСТ 1: Перевод одиночного текста ===")
    russian_phrase = "Привет! Это тестовая запись для проверки системы переозвучки видео."
    
    en_translation = translate_text(russian_phrase, target_lang="en")
    print(f"Оригинал (RU):  {russian_phrase}")
    print(f"Перевод (EN):   {en_translation}")

    es_translation = translate_text(russian_phrase, target_lang="es")
    print(f"Перевод (ES):   {es_translation}")

    print("\n=== ТЕСТ 2: Перевод сегментов с сохранением таймкодов ===")
    mock_whisper_segments = [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "Привет всем!"},
        {"id": 1, "start": 2.5, "end": 5.0, "text": "Сегодня мы создаем сервис переозвучки видео."},
    ]

    translated_segments = translate_segments(mock_whisper_segments, target_lang="en")

    print("\nРезультат обработки сегментов:")
    for seg in translated_segments:
        print(f"  [{seg['start']}s -> {seg['end']}s]")
        print(f"    Оригинал: {seg['original_text']}")
        print(f"    Перевод:  {seg['translated_text']}")


if __name__ == "__main__":
    run_test()