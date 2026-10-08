from pathlib import Path
from typing import Union
import asyncio
import edge_tts
import logging

logger = logging.getLogger(__name__)

VOICE_ACTORS = {
    "en": {
        "male": "en-US-ChristopherNeural",
        "female": "en-US-JennyNeural"       
    },
    "es": {
        "male": "es-ES-AlvaroNeural",
        "female": "es-ES-ElviraNeural"
    },
    "de": {
        "male": "de-DE-ConradNeural",
        "female": "de-DE-KatjaNeural"
    },
    "fr": {
        "male": "fr-FR-HenriNeural",
        "female": "fr-FR-DeniseNeural"
    },
    "ru": {
        "male": "ru-RU-DmitryNeural",
        "female": "ru-RU-SvetlanaNeural"
    }
}


async def _generate_edge_tts(text: str, voice_name: str, output_path: Path):
    """Асинхронная генерация нейро-голоса."""
    communicate = edge_tts.Communicate(text=text, voice=voice_name)
    await communicate.save(str(output_path))


def synthesize_speech(
    text: str, 
    output_audio_path: Union[str, Path], 
    language: str = "en",
    gender: str = "male"
) -> Path:
    """
    Синтезирует кинематографичную речь нужного пола и языка через нейросеть Edge-TTS.
    """
    if not text or not text.strip():
        raise ValueError("Текст для синтеза речи пуст.")

    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    lang_code = language.lower().strip()
    gender_code = gender.lower().strip() if gender in ["male", "female"] else "male"

    lang_voices = VOICE_ACTORS.get(lang_code, VOICE_ACTORS["en"])
    chosen_voice = lang_voices.get(gender_code, lang_voices["male"])

    logger.info(f"Нейро-синтез речи [{chosen_voice}]: '{text}'")

    try:
        asyncio.run(_generate_edge_tts(text.strip(), chosen_voice, output_audio_path))
        return output_audio_path

    except Exception as e:
        logger.error(f"Сбой нейро-синтеза Edge-TTS: {e}")
        raise RuntimeError(f"Ошибка синтеза речи: {e}")