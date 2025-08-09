import torch
import soundfile as sf
import simpleaudio as sa
import numpy as np
from pathlib import Path

# Import OpenVoice modules — adjust based on OpenVoice repo structure!
from openvoice import TTSBaseModel, ToneColorModel, Vocoder

REFERENCE_AUDIO = "reference.wav"
OUTPUT_AUDIO = "output.wav"

# You must set these paths to where you saved the OpenVoice checkpoints
TTS_BASE_CHECKPOINT = "checkpoints/tts_base.pth"
TONE_COLOR_CHECKPOINT = "checkpoints/tone_color.pth"
VOCODER_CHECKPOINT = "checkpoints/vocoder.pth"

def load_models():
    print("Loading models...")
    tts_model = TTSBaseModel.load_from_checkpoint(TTS_BASE_CHECKPOINT)
    tone_color_model = ToneColorModel.load_from_checkpoint(TONE_COLOR_CHECKPOINT)
    vocoder_model = Vocoder.load_from_checkpoint(VOCODER_CHECKPOINT)
    return tts_model, tone_color_model, vocoder_model

def extract_style_embedding(tone_color_model, ref_audio_path):
    print(f"Extracting style embedding from {ref_audio_path} ...")
    audio, sr = sf.read(ref_audio_path)
    # OpenVoice usually expects mono float32 numpy arrays
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    style_embedding = tone_color_model.extract_style(audio, sr)
    return style_embedding

def synthesize(tts_model, style_embedding, vocoder_model, text, output_path):
    print(f"Synthesizing text: {text}")
    # Synthesize mel spectrogram or audio waveform conditioned on style_embedding
    mel = tts_model.infer(text, style_embedding)
    
    # Vocoder generates waveform audio from mel spectrogram
    audio_waveform = vocoder_model.generate_waveform(mel)
    
    # Save output audio as 16kHz 16-bit wav
    sf.write(output_path, audio_waveform, 16000)
    print(f"Saved synthesized audio to {output_path}")

def play_audio(wav_path):
    print(f"Playing audio: {wav_path}")
    wave_obj = sa.WaveObject.from_wave_file(wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def main():
    if not Path(REFERENCE_AUDIO).exists():
        print(f"Reference audio file '{REFERENCE_AUDIO}' not found.")
        return

    # Load models
    tts_model, tone_color_model, vocoder_model = load_models()

    # Extract style from reference
    style_embedding = extract_style_embedding(tone_color_model, REFERENCE_AUDIO)

    # Synthesize speech
    synthesize(tts_model, style_embedding, vocoder_model, "Hello! This is a test of OpenVoice TTS.", OUTPUT_AUDIO)

    # Play audio
    play_audio(OUTPUT_AUDIO)

if __name__ == "__main__":
    main()
