import whisper
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
_MODEL_CACHE = {}
def _load_whisper(model_name: str):
    if model_name not in _MODEL_CACHE:
        print(f"[INFO] Loading Whisper model: {model_name} (this may take a moment)...")
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]

def transcribe_audio(audio_path: str, model_size: str = "small", force_language: str = None):
    model_name = model_size
    if force_language:
        if force_language.lower() == "en":
            model_name = f"{model_size}.en"

    model = _load_whisper(model_name)
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    whisper_lang = result.get("language", None)

    detected_lang = None
    try:
        if text and len(text) >= 6: 
            detected_lang = detect(text)
    except Exception:
        detected_lang = None
    if not detected_lang and whisper_lang:
        detected_lang = whisper_lang

    return text, detected_lang, whisper_lang
