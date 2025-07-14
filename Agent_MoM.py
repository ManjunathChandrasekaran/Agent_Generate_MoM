import os
import logging
import torch
from faster_whisper import WhisperModel
from llama_cpp import Llama
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transcribe_audio(audio_path, model_size="medium", language="en"):
    """Transcribe audio file using faster-whisper."""
    try:
        logger.info(f"Loading Whisper model: {model_size}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(model_size, device=device, compute_type="int8")

        logger.info(f"Transcribing audio file: {audio_path}")
        segments, info = model.transcribe(audio_path, language=language, beam_size=5)

        transcription = " ".join([segment.text for segment in segments])

        logger.info("Transcription completed successfully")
        return transcription.strip(), info.language
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise


def summarize_text(text, llm):
    """Summarize text using Llama-3.1-8B-Instruct for MoM."""
    try:
        logger.info("Generating Minutes of Meeting with Llama-3.1-8B-Instruct")
        prompt = f"""
You are an expert at creating concise and professional Minutes of Meeting (MoM). Summarize the following meeting transcript into a structured MoM with a title, date, time, attendees (if identifiable), key discussion points, decisions, and action items. Ensure the output is clear, professional, and formatted as shown below.

**Transcript**: "{text}"

**Minutes of Meeting**

**Meeting Title:** [Generate a suitable title based on content]  
**Date and Time:** {datetime.now().strftime('%Y-%m-%d %I:%M %p IST')}  
**Attendees:** [Infer names or roles if mentioned in transcript, or state 'Team Members' if unclear]  
**Summary of Discussion:**
- [Key point 1]
- [Key point 2]
**Decisions Made:**
- [Decision 1]
- [Decision 2]
**Action Items:**
- [Action item 1: Assigned to whom, deadline if specified]
- [Action item 2: Assigned to whom, deadline if specified]
"""

        # Correct llama-cpp-python usage (use __call__)
        response = llm(
            prompt=prompt,
            max_tokens=800,
            stop=["</s>"],
            temperature=0.3,
            top_p=0.9
        )

        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise


def save_mom(summary, output_file="minutes_of_meeting.txt"):
    """Save the MoM to a text file."""
    try:
        logger.info(f"Saving Minutes of Meeting to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(summary)
        logger.info("Minutes of Meeting saved successfully")
    except Exception as e:
        logger.error(f"Error saving Minutes of Meeting: {str(e)}")
        raise


def main(audio_path, output_file="minutes_of_meeting.txt"):
    """Main function to process audio, transcribe, summarize, and save MoM."""
    try:
        # Step 1: Transcribe audio
        transcription, detected_language = transcribe_audio(audio_path, model_size="medium", language="en")
        logger.info(f"Detected language: {detected_language}")
        logger.info(f"Transcription: {transcription[:100]}...")  # Log first 100 chars

        save_transcription(transcription, output_file="./output/raw_transcription.txt")

        # Step 2: Load Llama-3.1-8B-Instruct
        model_path = r"C:\Manju\LLM_Models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Llama model not found at: {model_path}")

        logger.info("Loading Llama-3.1-8B-Instruct model")
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=10,
            n_threads=max(os.cpu_count() - 1, 1)
        )

        # Step 3: Summarize transcription
        summary = summarize_text(transcription, llm)

        return summary
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

def save_transcription(transcription, output_file):
    """Save the raw transcription to a text file."""
    try:
        logger.info(f"Saving transcription to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(transcription)
        logger.info("Transcription saved successfully")
    except Exception as e:
        logger.error(f"Error saving transcription: {str(e)}")
        raise

def save_mom(summary, output_file):
    """Save the MoM to a text file."""
    try:
        logger.info(f"Saving Minutes of Meeting to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(summary)
        logger.info("Minutes of Meeting saved successfully")
    except Exception as e:
        logger.error(f"Error saving Minutes of Meeting: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    audio_file = "./SampleMeeting/meeting-clip1.mp3"  # Replace with your audio file path
    output_file = "./output/minutes_of_meeting.txt"
    if not os.path.exists(audio_file):
        logger.error(f"Audio file {audio_file} not found")
        raise FileNotFoundError(f"Audio file {audio_file} not found")

    summary = main(audio_file)
    print(summary)
    save_mom(summary, output_file)