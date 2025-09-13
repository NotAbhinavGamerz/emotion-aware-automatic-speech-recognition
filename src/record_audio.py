#for recording on terminal and testing
#not used in final app

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import re
import keyboard  

def get_next_filename(folder="data", prefix="Audio", extension=".wav"):
    os.makedirs(folder, exist_ok=True)
    existing_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(extension)]
    if not existing_files:
        return os.path.join(folder, f"{prefix}1{extension}")
    numbers = []
    for f in existing_files:
        match = re.search(rf"{prefix}(\d+){extension}", f)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(folder, f"{prefix}{next_num}{extension}")


def record_audio_dynamic(sample_rate=16000):
    print("🎤 Press 'r' to start recording...")
    keyboard.wait("r") 
    print("⏺ Recording... Press 's' to stop.")

    # Start recording
    recording = []
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()

    while True:
        data, _ = stream.read(1024)
        recording.append(data)
        if keyboard.is_pressed("s"): 
            break

    stream.stop()
    stream.close()
    print("Recording stopped.")
    recording = np.concatenate(recording, axis=0)
    filename = get_next_filename()
    wav.write(filename, sample_rate, recording)
    print(f"Saved as {filename}")
    return filename


if __name__ == "__main__":
    record_audio_dynamic()
