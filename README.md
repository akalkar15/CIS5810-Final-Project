# AI Video Summarization
CIS 5810 Final Project, dev by Tim Kao and Ankita Kalkar
## Prerequisites
 - python3.X (should be a Python version between 3.8 - 3.11 to avoid dependency conflicts!)
 - ffmpeg
   - Mac installation: `brew install ffmpeg`
 - GCP Speech API service account key, named `cis5810-speech-sa-key.json`
## Setup
 1. `cd src/`
    1. (Optional) Create virtual env: `python3 -m venv venv`
    2. `source venv/bin/activate`
 2. `pip install -r requirements.txt`
 3. `python main.py`
