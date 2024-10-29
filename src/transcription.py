import subprocess
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from pydub import AudioSegment

credentials = service_account.Credentials.from_service_account_file(
    'cis5810-speech-sa-key.json'
)

client = speech.SpeechClient(credentials=credentials)

def mp4_to_wav(file_path):
    file_path = file_path.replace(".mp4", "")
    subprocess.run(["ffmpeg", "-i", f"{file_path}.mp4", "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", f"{file_path}.wav"])
    print(f"Successfully converted {file_path}.mp4 to {file_path}.wav")
    return f"{file_path}.wav"

def transcribe(wav_path):
    """Invokes GCP Speech API to transcribe an audio into text.

    Args:
        wav_path (str): path to .wav file

    Returns:
        str: string transcription of the audio file
    """
    print("Transcribing audio...")
    audio_file = wav_path

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
    print("Finished transcribing audio")
    return ". ".join([r.alternatives[0].transcript for r in response.results])
