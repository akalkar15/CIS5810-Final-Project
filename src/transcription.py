import subprocess
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from pydub import AudioSegment
import os
import librosa
import numpy as np

credentials = service_account.Credentials.from_service_account_file(
    'cis5810-speech-sa-key.json'
)

client = speech.SpeechClient(credentials=credentials)

def mp4_to_wav(file_path):
    file_path = file_path.replace(".mp4", "")
    subprocess.run(["ffmpeg", "-i", f"{file_path}.mp4", "-ab", "160k", "-ac", "2", "-ar", "44100", "-loglevel", "error", "-stats","-vn", f"{file_path}.wav"])
    print(f"Successfully converted {file_path}.mp4 to {file_path}.wav")
    return f"{file_path}.wav"

def transcribe(file_path):
    """Invokes GCP Speech API to transcribe an audio into text.

    Args:
        wav_path (str): path to .wav file

    Returns:
        str: string transcription of the audio file
    """
    print("Transcribing audio...")
    audio_file  = mp4_to_wav(file_path)

    # Convert stereo to mono using pydub
    sound = AudioSegment.from_wav(audio_file)
    sound = sound.set_channels(1)  # Convert to mono
    mono_audio_file = f"{audio_file.replace('.wav', '')}_mono.wav"
    sound.export(mono_audio_file, format="wav")

    with open(mono_audio_file, 'rb') as audio:
        content = audio.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US'
    )

    response = client.recognize(config=config, audio=audio)
    dialogue = ". ".join([r.alternatives[0].transcript for r in response.results if r.alternatives])

    y, sr = librosa.load(audio_file)
    
    # Extract tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract key and mode
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Average chroma across time
    chroma_avg = np.mean(chroma, axis=1)
    
    # Determine the pitch class with the highest energy
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_pitch = pitch_classes[np.argmax(chroma_avg)]

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    mode = "minor" if np.mean(tonnetz) < 0 else "major"
    
    # Analyze dynamics
    rms = librosa.feature.rms(y=y).mean()
    dynamics = "soft" if rms < 0.02 else "loud"
    
    # Instrumentation analysis (requires a more advanced library like Essentia)
    instrumentation = "orchestral"  # Placeholder
    
    music = {
        "tempo": round(tempo[0]),
        "key": detected_pitch,
        "mode": mode,
        "dynamics": dynamics,
        "instrumentation": instrumentation,
    }
    print(f"Finished transcribing audio for scene {file_path}")

    return dialogue, music