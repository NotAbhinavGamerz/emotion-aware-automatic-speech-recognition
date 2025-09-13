import argparse,sys,os

sys.path.append(os.path.dirname(__file__))

from asr_whisper import transcribe_audio
from emotion_model import detect_emotion

try:
    from record_audio import record_audio_dynamic
except ImportError:
    record_audio_dynamic = None

def run_pipeline(audio_file, model_size="small", force_lang=None, debug_emo=False):
    print("\nRunning Emotion-Aware ASR")
    transcript, detected_lang, whisper_lang = transcribe_audio(audio_file, model_size=model_size, force_language=force_lang)
    print(f"[ASR] Transcript: {transcript}")
    print(f"[ASR] Detected language (text-based): {detected_lang} | Whisper: {whisper_lang}")

    final_audio_label, audio_score, audio_predictions, final_text_label, text_score, text_predictions = detect_emotion(audio_file, transcript, debug=debug_emo)
    print("\n[EMO] Audio Emotion:")
    print(f"  {final_audio_label} ({audio_score:.2f})")

    print("\n[EMO] Text Emotion:")
    print(f"  {final_text_label} ({text_score:.2f})")

    print("\nFinal Annotated Transcript:")
    print(f"{transcript} [{final_audio_label} / {final_text_label}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["file", "record"], default="file")
    parser.add_argument("--audio_file", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--force_lang", type=str, default=None)
    parser.add_argument("--debug_emo", action="store_true")
    args = parser.parse_args()

    if args.mode == "file":
        if not args.audio_file:
            print("Provide --audio_file when using mode=file")
            sys.exit(1)
        run_pipeline(
            args.audio_file,
            model_size=args.model_size,
            force_lang=args.force_lang,
            debug_emo=args.debug_emo,
        )
    else:
        if record_audio_dynamic is None:
            print("Recording not available (missing record_audio.py)")
            sys.exit(1)
        audio_file = record_audio_dynamic()
        run_pipeline(
            audio_file,
            model_size=args.model_size,
            force_lang=args.force_lang,
            debug_emo=args.debug_emo,
        )
