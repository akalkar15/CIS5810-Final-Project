import supervision as sv
from ultralytics import YOLO
import numpy as np
from scenedetect import detect, AdaptiveDetector, video_splitter
from collections import Counter

import os
import shutil
import logging

# Set ultralytics logging level to WARNING or ERROR
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("supervision").setLevel(logging.WARNING)

# Load YOLO model
model = YOLO('yolov8n.pt')

def split_scenes(scene_path, out_dir="/"):
    """Splits an input video into scenes and outputs them in directory specified

    Args:
        scene_path (_type_): _description_
        out_dir (str, optional): _description_. Defaults to '/'.
    Returns:
        scene_list: list of scene names
    """
    print("Splitting scenes...")
    if os.listdir(out_dir):
        shutil.rmtree(out_dir)
    scene_list = detect(scene_path, AdaptiveDetector())
    video_splitter.split_video_ffmpeg(
        input_video_path=scene_path, 
        scene_list=scene_list,  
        show_progress=True,
        output_dir=out_dir,
        show_output=False
    )
    time_splits = [(start.get_timecode(), end.get_timecode()) for (start, end) in scene_list]
    print(f"Successfully generated {len(scene_list)} scenes in directory: '{out_dir}'")
    return time_splits, sorted(os.listdir(out_dir))

def detect_objects(scene_path):
    print("Detecting objects...")
    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    raw_detections = []

    def forward_fuse(self, x):
        """Apply fused convolution and activation to input tensor."""
        return self.act(self.conv(x))
    
    # Callback function to process each frame
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        raw_detections.append(detections)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame

    # Process video and save output
    sv.process_video(
        source_path=scene_path,
        target_path=f"{scene_path.replace('.mp4', '')}-tracked.mp4",
        callback=callback
    )
    print(f"Finished processing object detections for scene {scene_path}")
    mapped_detections = [f"{model.model.names[class_id]}{tid}" for raw in raw_detections for class_id, tid in zip(raw.class_id, raw.tracker_id)]
    counter = Counter(mapped_detections)
    total_count = len(mapped_detections)
    threshold = 0.08
    # print(set(mapped_detections))
    # Filter the strings with more than 10% prevalence
    filtered_detections = [string for string in mapped_detections if counter[string] / total_count > threshold]
    print(f"Objects for scene {scene_path}: ", set(filtered_detections))
    return set(filtered_detections)

if __name__ == "__main__":
    detect_objects("data/scenes/tangled-Scene-016.mp4")
    