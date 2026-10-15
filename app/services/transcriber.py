import re
from pathlib import Path
from typing import Union, Dict, Any
import whisper
import static_ffmpeg
import logging

static_ffmpeg.add_paths()
logger = logging.getLogger(__name__)

_LOADED_MODELS: Dict[str, whisper.Whisper] = {}

SOUND_TAGS = [
    "СТУК В ДВЕРЬ", "СТУК", "ВЫСТРЕЛЫ", "ВЫСТРЕЛ", "МУЗЫКА", 
    "ХЛОПОК", "ВЗДОХ", "ТИШИНА", "КРИК", "ШАГИ", "АПЛОДИСМЕНТЫ", "ШУМ", "ВЗРЫВ"
]


def get_whisper_model(model_size: str = "base") -> whisper.Whisper:
    if model_size not in _LOADED_MODELS:
        logger.info(f"Загрузка модели Whisper '{model_size}'...")
        _LOADED_MODELS[model_size] = whisper.load_model(model_size)
    return _LOADED_MODELS[model_size]


def is_noise_or_loop(text: str) -> bool:
    clean_upper = re.sub(r'[^\w\s]', '', text).upper().strip()
    if clean_upper in SOUND_TAGS or re.search(r'(.)\1{4,}', text):
        return True
    return False


def transcribe_audio(
    audio_path: Union[str, Path], 
    model_size: str = "base",
    source_language: str = "auto"
) -> Dict[str, Any]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    model = get_whisper_model(model_size)
    
    lang_param = None if source_language == "auto" else source_language
    logger.info(f"Транскрибация файла ({audio_path.name}) с языком оригинала: {source_language.upper()}")

    result = model.transcribe(
        str(audio_path),
        fp16=False,
        language=lang_param,
        word_timestamps=True
    )

    detected_lang = result.get("language", source_language if source_language != "auto" else "en")
    if detected_lang in ["haw", "jw", "la", "sn", "su"]:
        logger.warning(f"Whisper ошибся языком ({detected_lang}). Принудительно выставляем русский.")
        detected_lang = "ru"
    logger.info(f"Язык транскрибации: {detected_lang.upper()}")

    clean_segments = []
    full_text_list = []

    for seg in result.get("segments", []):
        raw_text = seg.get("text", "").strip()

        if not raw_text or is_noise_or_loop(raw_text):
            continue

        words = seg.get("words", [])
        if words:
            exact_start = round(words[0]["start"], 2)
            exact_end = round(words[-1]["end"], 2)
        else:
            exact_start = round(seg["start"], 2)
            exact_end = round(seg["end"], 2)

        clean_segments.append({
            "id": seg["id"],
            "start": exact_start,
            "end": exact_end,
            "duration": round(exact_end - exact_start, 2),
            "text": raw_text
        })
        full_text_list.append(raw_text)

    full_clean_text = " ".join(full_text_list).strip()

    return {
        "text": full_clean_text,
        "language": detected_lang,
        "segments": clean_segments
    }