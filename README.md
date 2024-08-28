# AI-Driven Meeting Minutes Generator
## Overview
The AI-Driven Meeting Minutes Generator is a comprehensive solution designed to automate the creation of meeting minutes from video recordings. By leveraging advanced machine learning models, this tool extracts audio from uploaded videos, transcribes the spoken content into text, and generates structured, professional meeting minutes that capture key discussion points, decisions, and action items.

## Flowchart

![WhatsApp Image 2024-08-27 at 22 55 50_1b6bdd0c](https://github.com/user-attachments/assets/24ecdea1-b5b0-4bbf-8891-31b0c9528d11)


## Features
--> Automatic Audio Extraction: Effortlessly extracts audio from video files, ensuring seamless processing. <br>
--> Accurate Speech-to-Text Conversion: Uses OpenAI's Whisper model for precise transcription with contextual awareness. <br>
--> Advanced Summarization: Fine-tuned RAG model identifies and synthesizes critical topics, decisions, and action items. <br>
--> Professional Document Output: Delivers well-organized, professionally formatted meeting minutes. <br>
--> Enhanced Meeting Efficiency: Streamlines documentation, improving overall meeting management. <br>

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/meeting-minutes-generator.git
    cd meeting-minutes-generator
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure FFmpeg is installed and accessible in your system's PATH. Follow the [FFmpeg installation guide](https://ffmpeg.org/download.html) for your operating system.

## Usage

1. **Upload a Video**: Place your meeting video file in the `input/` directory.

2. **Run the Generator**:
    ```bash
    python generate_minutes.py --input input/meeting_video.mp4 --output output/meeting_minutes.txt
    ```

3. **Output**: The generated meeting minutes will be saved in the `output/` directory as a text file.
