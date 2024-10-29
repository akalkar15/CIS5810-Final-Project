# AI Video Summarization
CIS 5810 Final Project, dev by Tim Kao and Ankita Kalkar
## Prerequisites
 - python3.X
 - ffmpeg
   - Mac installation: `brew install ffmpeg`
 - GCP Speech API service account key, named `cis-speech-sa-key.json`
## Setup
 1. `cd src/`
    1. (Optional) Create virtual env: `python3 -m venv venv`
    2. `source venv/bin/activate`
 2. `pip install -r requirements.txt`
 3. `python main.py`