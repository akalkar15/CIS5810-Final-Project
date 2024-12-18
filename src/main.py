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
from collections import defaultdict
import easygui

os.path.join(os.getcwd(), 'src')

def time_to_seconds(time_str):
    """Convert a time string (hh:mm:ss.sss) to total seconds."""
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

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
        asyncio.to_thread(transcribe, f"data/scenes/{scene}"),
    ]

    # Wait for all three results for the file
    results = await asyncio.gather(*futures)

    # Store results in the dictionary
    result = {}
    if results[0]:
        result['detections'] = results[0]
    if results[1]:
        result['lighting_analysis'] = results[1]
    if results[2]:
        result['face_analysis'] = results[2]
    if results[3][0]:
        result['transcription'] = results[3][0]
    if results[3][1]:
        result['soundtrack_analysis'] = results[3][1]
        print(results[3][1])

    scene_num = int(scene[-7:-4])
    summary = retrieve_summary(result)
    return summary, scene_num, results[3]

async def process_video(filepath):
    if '/' in filepath:
        file_name = filepath.split('/')[2].replace('.mp4', '')
    else:
        file_name = filepath.replace('.mp4', '')
    time_splits, scene_list = split_scenes(filepath, "data/scenes/") # list of scene names, numbered starting at 001
    scene_durations = [time_to_seconds(end) - time_to_seconds(start) for start, end in time_splits]

    # Normalize weights
    total_duration = sum(scene_durations)
    scene_weights = [duration / total_duration for duration in scene_durations]
    scene_data = {}

    # Process scenes in parallel
    tasks = [process_single_scene(scene) for scene in scene_list]
    results = await asyncio.gather(*tasks)

    scene_data = []
    transcriptions = defaultdict(int)
    for (time_range, (summary, scene_num, transcription), weight) in zip(time_splits, results, scene_weights):
        item = {}
        item['summary'] = summary
        item['time_range'] = time_range
        item['weight'] = weight
        scene_data.append(item)
        transcriptions[scene_num] = transcription  # Store the transcription

    with open("scene_data.json", "w+") as f:
        f.write(json.dumps(scene_data))

    # Call GPT for summary
    overall_summary = combine_summaries(scene_data, transcriptions)
    print("OVERALL SUMMARY: \n", overall_summary)
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

    summary = asyncio.run(process_video(filepath))
    easygui.msgbox(summary, title="Summary of Scene")
