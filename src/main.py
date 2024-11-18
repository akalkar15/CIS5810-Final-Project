from scene_detector import split_scenes, detect_objects
from transcription import mp4_to_wav, transcribe
from light_analysis import analyze_lighting_and_color
from facial_expression import get_facial_attributes
from flow import get_direction
from openai_helper import retrieve_summary, combine_summaries
import argparse
import multiprocessing
import asyncio
import os
import json

os.path.join(os.getcwd(), 'src')

async def process_single_scene(scene):
    # Run three synchronous functions in parallel for each file
    # futures = [
    #     loop.run_in_executor(executor, detect_objects(f"data/scenes/{scene}")),
    #     loop.run_in_executor(executor, analyze_lighting_and_color(f"data/scenes/{scene}"))
    # ]
    futures = [
        asyncio.to_thread(detect_objects, f"data/scenes/{scene}"),
        asyncio.to_thread(analyze_lighting_and_color, f"data/scenes/{scene}"),
        asyncio.to_thread(get_facial_attributes, f"data/scenes/{scene}"),
        asyncio.to_thread(transcribe, f"data/scenes/{scene}")
    ]

    # Wait for all three results for the file
    results = await asyncio.gather(*futures)

    # Store results in the dictionary
    result = {
        "detections": results[0],
        "lighting_analysis": results[1],
        "face_analysis": results[2],
        "transcription": results[3],
    }
    summary = retrieve_summary(result)
    return summary

async def process_video(filepath):
    if '/' in filepath:
        file_name = filepath.split('/')[2].replace('.mp4', '')
    else:
        file_name = filepath.replace('.mp4', '')
    time_splits, scene_list = split_scenes(filepath, "data/scenes/") # list of scene names, numbered starting at 001

    scene_data = {}

    # Process scenes in parallel
    tasks = [process_single_scene(scene) for scene in scene_list]
    results = await asyncio.gather(*tasks)

    scene_data = {str(time_range): result for time_range, result in zip(time_splits, results)}
    
    with open("scene_data.json", "w+") as f:
        f.write(json.dumps(scene_data))

    # Call GPT for summary
    overall_summary = combine_summaries(scene_data)
    return overall_summary

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
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
    print(asyncio.run(process_video(filepath)))