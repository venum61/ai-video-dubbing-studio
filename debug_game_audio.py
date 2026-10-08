from pathlib import Path
import whisper
import static_ffmpeg

static_ffmpeg.add_paths()

audio_files = list(Path("storage/temp").glob("extracted_*.wav"))
if not audio_files:
    print("Файл не найден в storage/temp")
    exit()

target_audio = audio_files[0]
print(f"Анализируем аудиофайл: {target_audio.name}")

model = whisper.load_model("base")
result = model.transcribe(str(target_audio), fp16=False, language="ru")

print("\n" + "=" * 50)
print("СЫРОЙ ТЕКСТ ОТ WHISPER (БЕЗ ФИЛЬТРОВ):")
print("=" * 50)
print(f"Полный текст: '{result['text']}'")
print("\nСегменты с таймкодами:")
for seg in result["segments"]:
    print(f"  [{seg['start']}s -> {seg['end']}s]: '{seg['text']}' (logprob: {seg['avg_logprob']:.2f}, no_speech: {seg['no_speech_prob']:.2f})")
print("=" * 50)