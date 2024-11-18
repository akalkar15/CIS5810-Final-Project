from openai import OpenAI

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
}

**Scene Data**
<<data>>

Summary:
"""

COMBINE_PROMPT = """**Task:**
You are provided summaries of a video split by scenes. Each scene is given by its start and end time and its summary. Using this information, combine all of the summaries into a single overall summary of the video's content. The style should be like a synopsis.
- Omit direct mentions of the provided data. Simply use them to describe what is happening in the scene.

Input Schema:
{
    (start_time, end_time): "<<short summary of the scene>>"
}

**Scene Data**
<<data>>

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

def combine_summaries(scene_data):
    print("Retrieving summary from OpenAI using model gpt-4o-mini...")
    prompt = COMBINE_PROMPT.replace('<<data>>', str(scene_data))
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
