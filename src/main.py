from scene_detector import split_scenes, detect_objects
from transcription import mp4_to_wav, transcribe
from light_analysis import analyze_lighting_and_color
from flow import get_direction
from openai_helper import retrieve_summary
import argparse
import os

os.path.join(os.getcwd(), 'src')

def process_video(filepath):
    if '/' in filepath:
        file_name = filepath.split('/')[2].replace('.mp4', '')
    else:
        file_name = filepath.replace('.mp4', '')
    scene_list = split_scenes(filepath, "data/scenes/") # list of scene names, numbered starting at 001
    detections = detect_objects(f"data/scenes/{file_name}-Scene-001.mp4") # TODO: process multiple scenes
    transcription = transcribe(mp4_to_wav(f"data/scenes/{file_name}-Scene-001.mp4")) # TODO: process multiple scenes
    analysis = analyze_lighting_and_color(f"data/scenes/{file_name}-Scene-001.mp4")
    direction = get_direction(f"data/scenes/{file_name}-Scene-001.mp4")
    #print(direction)
    summary = retrieve_summary(detections, analysis, transcription)
    return summary

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
    print(process_video(filepath))