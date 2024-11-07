from openai import OpenAI

client = OpenAI()

PROMPT = """**Task:**
Using the provided object detections, lighting and color description, and the audio transcription, provide a detailed summary of the video's content.
- Omit direct mentions of the provided data. Simply use them to describe what is happening in the scene.

**Detections**
<<detections>>

**Transcription**
<<transcription>>

The lighting and color in the scene provides a <<lighting>> mood.

Summary:
"""

def retrieve_summary(detections, lighting_analysis, transcription):
    print("Retrieving summary from OpenAI using model gpt-4o...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are an intelligent analyst trained on video summarization."},
            {
                "role": "user",
                "content": PROMPT.replace('<<detections>>', str(detections)).replace('<<transcription>>', transcription).replace('<<lighting>>', lighting_analysis)
            }
        ]
    )
    return completion.choices[0].message.content
