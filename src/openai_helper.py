from openai import OpenAI
import os

client = OpenAI()


SINGLE_SCENE_PROMPT = """**Task:**
You are provided information about a scene as a json object. The information includes object detections, lighting and color description, facial attributes, and the audio transcription. Using this information, provide a detailed overall summary of the video's content. Be descriptive, especially if there is information about people in the scene.
- Omit direct mentions of the provided data. Simply use them to describe what is happening in the scene.

Input Schema:
{
    "detections": [<<list of object detetions within the scene],
    "transcription": "<<text-to-speech transcription of the audio>>",
    "lighting_analysis": "<<few words describing the lighting and color of the scene>>",
    "face_analysis": "<<dictionary of face attributes, with keys indicating ids of each identified person>>"
    "soundtrack_analysis": {
        "tempo": <<tempo in beats per minute>>,
        "key": "<<key of the soundtrack>>",
        "mode": "<<major or minor>>",
        "dynamics": "<<soft or loud>>",
        "instrumentation": "<<type of instruments or style of the music>>"
    }

}

**Scene Data**
<<data>>

Summary:
"""

COMBINE_PROMPT = """**Task:**
You are provided summaries of a video split by scenes. Each scene is given by its start and end time, its summary, a weight that indicates its importance, and the full transcription of the video. Using this information, combine all the summaries using the data provided into a single overall summary of the video's content. The style should be like a synopsis.
- Heavily emphasize the summaries with higher weights.
- Use the provided weights to determine the importance of each scene in the overall summary.
- Omit direct mentions of the provided data. Simply use them to describe what is happening in the scene.
- Use as much detail as possible and keep details about the dialogue

Input Schema:
{
    (start_time, end_time): "<<short summary of the scene>> (Weight: <<weight>>)"
}

**Scene Data**
<<data>>

**Full Transcription**
<<transcription>>

Summary:
"""

def retrieve_summary(scene_data):
    print("Retrieving summary from OpenAI using model gpt-4o...")
    prompt = SINGLE_SCENE_PROMPT.replace('<<data>>', str(scene_data))
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are an intelligent analyst trained on video summarization."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    if not completion:
        print("Error retrieving summary from OpenAI")
        return ""
    return completion.choices[0].message.content

def combine_summaries(scene_data, transcription):
    print("Retrieving summary from OpenAI using model gpt-4o-mini...")
    weighted_data = {}
    for scene in scene_data:
        start, end = scene["time_range"]
        summary = scene["summary"]
        weight = scene["weight"]
        # Include weights in the input data for OpenAI
        weighted_data[(start, end)] = f"{summary} (Weight: {weight:.2f})"

    # Replace in prompt
    formatted_data = str(weighted_data).replace("'", '"')  # JSON-like formatting for clarity
    formatted_transcription = str(transcription).replace("'", '"')
    prompt = COMBINE_PROMPT.replace("<<data>>", formatted_data)
    prompt = COMBINE_PROMPT.replace("<<transcription>>", formatted_transcription)
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are an intelligent analyst trained on video summarization."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    if not completion:
        print("Error retrieving summary from OpenAI")
        return ""
    return completion.choices[0].message.content
