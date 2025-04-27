# Audio Analysis Module

This is an audio analysis module that can transcribe uploaded audio files into text and detect harmful content within the text.

## Features

- Supports uploading audio files in `.wav` and `.mp3` formats
- Uses OpenAI Whisper model for speech-to-text conversion
- Supports transcription of both Chinese and English audio
- Uses Huggingface model for harmful content detection
- Provides detailed harmful content classification results
- Clean and intuitive Streamlit user interface

## Installation Instructions

### System Requirements

- Python 3.8 or higher
- Sufficient disk space for storing models (may require several hundred MB to several GB depending on selected model size)
- Recommended to use CUDA-enabled GPU for accelerated processing (optional)

### Installation Steps

1. Clone or download this project locally

2. Create and activate a virtual environment (optional but recommended)

```bash
# Using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage Instructions

1. Launch the application

```bash
streamlit run app.py
```

2. Open the application in your browser (usually opens automatically at http://localhost:8501)

3.Usage workflow:
- Select model and settings in the sidebar
- Click the "Load Model" button
- Upload an audio file (.wav or .mp3 format)
- Click the "Start Transcription and Analysis" button
- View transcription results and harmful content detection results

## Supported Languages

The current version supports audio transcription and harmful content detection for the following languages:

- Chinese
- English

## Future Work

Planned support for the following UN official languages in future versions:

- French
- Spanish
- Arabic
- Russian

The code already includes interfaces for these languages, but the current version does not yet implement complete harmful content detection functionality.

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── audio_utils.py      # Audio processing module
├── text_utils.py       # Text processing and harmful content detection module
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── samples/            # Sample audio files directory (optional). Initialize oneself to establish.
```

## Notes

- On first run, the program will automatically download required models, which may take some time
- Processing large audio files may take considerable time, especially without GPU acceleration
- Harmful content detection accuracy depends on the selected model and threshold settings
