from faster_whisper import WhisperModel

# Load model once
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

async def transcribe_audio(audio_path: str) -> str:
    try:
        segments, _ = whisper_model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")