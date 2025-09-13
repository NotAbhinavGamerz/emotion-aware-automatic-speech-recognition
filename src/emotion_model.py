from transformers import pipeline

# Audio model
AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-er"
print(f"[INFO] Loading audio emotion model: {AUDIO_MODEL_NAME}")
audio_classifier = pipeline("audio-classification", model=AUDIO_MODEL_NAME)
print("[INFO] Audio emotion model loaded")

# Text model
TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
print(f"[INFO] Loading text emotion model: {TEXT_MODEL_NAME}")
text_classifier = pipeline("text-classification", model=TEXT_MODEL_NAME, return_all_scores=True)
print("[INFO] Text emotion model loaded")

# Map audio labels to full names
audio_label_map = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}

def _intensity_label(base_label: str, score: float) -> str:
    if score >= 0.85:
        return f"strongly {base_label}"
    elif score >= 0.65:
        return f"moderately {base_label}"
    else:
        return f"slightly {base_label}"

def detect_emotion(audio_path: str, transcript: str, debug: bool = False):
    # Audio prediction
    audio_results = audio_classifier(audio_path, top_k=3)
    audio_predictions = []
    for r in audio_results:
        lbl = audio_label_map.get(r["label"], r["label"])
        audio_predictions.append((lbl, float(r["score"])))
    audio_predictions.sort(key=lambda x: x[1], reverse=True)
    top_audio_label, top_audio_score = audio_predictions[0]
    if len(audio_predictions) > 1 and abs(audio_predictions[0][1] - audio_predictions[1][1]) < 0.15:
        final_audio_label = f"uncertain ({audio_predictions[0][0]}/{audio_predictions[1][0]})"
    else:
        final_audio_label = _intensity_label(top_audio_label, top_audio_score)

    # Text prediction
    text_results = text_classifier(transcript)
    all_scores = text_results[0]
    text_predictions = []
    for entry in all_scores:
        lbl = entry["label"]
        score = entry["score"]
        text_predictions.append((lbl, float(score)))
    text_predictions.sort(key=lambda x: x[1], reverse=True)
    top_text_label, top_text_score = text_predictions[0]

    if debug:
        print("[DEBUG] Audio predictions:")
        for lbl, sc in audio_predictions:
            print(f"  {lbl}: {sc:.2f}")
        print("[DEBUG] Text predictions:")
        for lbl, sc in text_predictions[:5]:
            print(f"  {lbl}: {sc:.2f}")

    return final_audio_label, top_audio_score, audio_predictions, top_text_label, top_text_score, text_predictions
