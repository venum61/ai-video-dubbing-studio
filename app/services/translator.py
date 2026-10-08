import re
from typing import List, Dict, Any
from deep_translator import GoogleTranslator, MyMemoryTranslator
import logging

logger = logging.getLogger(__name__)

LOCALE_MAP = {
    "ru": "ru-RU",
    "en": "en-US",
    "es": "es-ES",
    "de": "de-DE",
    "fr": "fr-FR",
    "it": "it-IT",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "uk": "uk-UA",
    "be": "be-BY",
    "kz": "kk-KZ"
}


def normalize_lang(code: str, default_fallback: str = "ru-RU") -> str:
    """
    Приводит короткий код языка к формату локали (например, 'ru' -> 'ru-RU').
    """
    if not code or code == "auto":
        return default_fallback
    
    code_lower = code.lower().strip()
    return LOCALE_MAP.get(code_lower, default_fallback)


def optimize_speech_phrasing(text: str) -> str:
    """
    Делает фразы короткими и разговорными, чтобы они идеально укладывались в хронометраж.
    """
    text = re.sub(r'\bHello,\s*my\s*name\s*is\b', "Hi, I'm", text, flags=re.IGNORECASE)
    text = re.sub(r'\bMy\s*name\s*is\b', "I am", text, flags=re.IGNORECASE)
    return text.strip()


def translate_text(
    text: str, 
    source_lang: str = "auto", 
    target_lang: str = "ru"
) -> str:
    """
    Отказоустойчивый перевод отдельной строки с автоматическим переключением провайдеров.
    """
    if not text or not text.strip():
        return ""

    optimized_text = optimize_speech_phrasing(text)

    try:
        src = source_lang if len(source_lang) == 2 else "auto"
        translator = GoogleTranslator(source=src, target=target_lang)
        return translator.translate(optimized_text)
    except Exception as e:
        logger.warning(f"Google Translate недоступен. Переключаемся на MyMemory...")

    try:
        src_locale = normalize_lang(source_lang, default_fallback="en-US" if target_lang == "ru" else "ru-RU")
        tgt_locale = normalize_lang(target_lang, default_fallback="ru-RU")
        
        fallback_translator = MyMemoryTranslator(source=src_locale, target=tgt_locale)
        return fallback_translator.translate(optimized_text)
    except Exception as fallback_error:
        logger.error(f"Сбой сервисов перевода: {fallback_error}")
        return optimized_text


def translate_segments(
    segments: List[Dict[str, Any]], 
    source_lang: str = "auto", 
    target_lang: str = "ru"
) -> List[Dict[str, Any]]:
    """
    Переводит список сегментов Whisper с сохранением исходных таймкодов.
    """
    if not segments:
        return []

    translated_segments = []
    logger.info(f"Перевод {len(segments)} сегментов: {source_lang.upper()} -> {target_lang.upper()}...")

    for seg in segments:
        orig = seg.get("text", "").strip()
        
        if orig:
            trans = translate_text(orig, source_lang=source_lang, target_lang=target_lang)
        else:
            trans = ""

        translated_segments.append({
            "id": seg.get("id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "duration": seg.get("duration", round(seg.get("end", 0) - seg.get("start", 0), 2)),
            "original_text": orig,
            "translated_text": trans
        })

    return translated_segments