from scene_detector import split_scenes, detect_objects
from transcription import mp4_to_wav, transcribe
import argparse
import os

os.path.join(os.getcwd(), 'src')

def process_video(filepath):
    file_name = filepath.replace('.mp4', '')
    scene_list = split_scenes(filepath, 'scenes/') # list of scene names, numbered starting at 001
    detections = detect_objects(f"scenes/{file_name}-Scene-001.mp4") # TODO: process multiple scenes
    # TODO: handle detection information, Detection Object API: https://supervision.roboflow.com/detection/core/#supervision.detection.core.Detections
    transcription = transcribe(mp4_to_wav(f"scenes/{file_name}-Scene-001.mp4")) # TODO: process multiple scenes
    print(transcription)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file information.")

    # Add arguments
    parser.add_argument("--filepath", type=str, help="path to video file to be processed", required=True)

    # Parse the arguments
    args = parser.parse_args()
    filepath = args.filepath

    # Use the arguments
    if not filepath or not os.path.exists(filepath):
        print("Missing or invalid required argument: --filepath")
        exit(1)
    process_video(filepath)