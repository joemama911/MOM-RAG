# MOM-RAG
this repo contains MOM / Notes generator from a meeting 

# AI-Driven Meeting Minutes Generator
## Overview
The AI-Driven Meeting Minutes Generator is a comprehensive solution designed to automate the creation of meeting minutes from video recordings. By leveraging advanced machine learning models, this tool extracts audio from uploaded videos, transcribes the spoken content into text, and generates structured, professional meeting minutes that capture key discussion points, decisions, and action items.

## Features
--> Automatic Audio Extraction: Effortlessly extracts audio from video files, ensuring seamless processing.
--> Accurate Speech-to-Text Conversion: Uses OpenAI's Whisper model for precise transcription with contextual awareness.
--> Advanced Summarization: Fine-tuned RAG model identifies and synthesizes critical topics, decisions, and action items.
--> Professional Document Output: Delivers well-organized, professionally formatted meeting minutes.
--> Enhanced Meeting Efficiency: Streamlines documentation, improving overall meeting management.

## Installation

## Prerequisites
Python 3.8 or higher
Required Python packages (listed in requirements.txt)
FFmpeg (for audio extraction)

## Setup: 
   git clone https://github.com/yourusername/meeting-minutes-generator.git
   cd meeting-minutes-generator


## Install the required dependencies:
   pip install -r requirements.txt
 Ensure FFmpeg is installed and accessible in your system's PATH. Follow the FFmpeg installation guide for your operating system.

## Usage
Upload a Video: Place your meeting video file in the input/ directory.

## Run the Generator:

python generate_minutes.py --input input/meeting_video.mp4 --output output/meeting_minutes.txt
Output: The generated meeting minutes will be saved in the output/ directory as a text file.
