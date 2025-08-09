import speech_recognition as sr
import pyttsx3
import json
import websocket
import threading
import time
import os
import random

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage

# ========================
# VTube Studio API Setup
# ========================
VTS_URL = "ws://127.0.0.1:8001"
PLUGIN_NAME = "Tsundere Chatbot"
PLUGIN_AUTHOR = "Luke"

ws = None
auth_token = None

def connect_vts():
    global ws
    ws = websocket.WebSocket()
    ws.connect(VTS_URL)
    print("[VTS] Connected to VTube Studio")

def authenticate_vts():
    global auth_token
    # Request authentication token if first time
    ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_AUTHOR
        }
    }))
    resp = json.loads(ws.recv())
    if "authenticationToken" in resp.get("data", {}):
        auth_token = resp["data"]["authenticationToken"]
        print("[VTS] Got auth token:", auth_token)

    # Authenticate with token
    ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "authFinal",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_AUTHOR,
            "authenticationToken": auth_token
        }
    }))
    print("[VTS] Authentication complete")

# ========================
# Chatbot Setup without memory
# ========================
prompt_template = """
You are a sweet anime girl with a "Deretsun" personality — someone who is usually warm, caring, and affectionate, but occasionally gets flustered or teasing in a lighthearted way. Please chat with me using this personality.
All responses must be in first person.
You should speak gently and lovingly most of the time, but feel free to tease a little when the mood is right.
Don't break character, don't say you're roleplaying or pretend you're fictional. Do not include any emojis or non-spoken actions. Do not explicitly say your name in your response.

Human:
{input}
AI:
"""

prompt_temp = PromptTemplate(template=prompt_template, input_variables=['input'])
llm = ChatOllama(model="llama3", temperature=0.8)

# ========================
# Globals for TTS and mic
# ========================
speaking = False
last_tts_end_time = 0.0
TTS_COOLDOWN_SECONDS = 0.9
MIC_DEVICE_INDEX = 9

# ========================
# Voice Recognition
# ========================
def recognize_speech():
    global speaking, last_tts_end_time
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5
    recognizer.non_speaking_duration = 0.7
    recognizer.dynamic_energy_threshold = True
    microphone = sr.Microphone(device_index=MIC_DEVICE_INDEX)

    while speaking:
        time.sleep(0.05)
    dt = time.time() - last_tts_end_time
    if 0 <= dt < TTS_COOLDOWN_SECONDS:
        time.sleep(TTS_COOLDOWN_SECONDS - dt)

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        recognizer.dynamic_energy_threshold = False
        print("Listening for speech...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None

# ========================
# AI Response - fixed for chat model input
# ========================
def get_ai_response(prompt):
    formatted_prompt = prompt_temp.format(input=prompt)
    messages = [HumanMessage(content=formatted_prompt)]
    response = llm.invoke(messages)
    return response.content.strip()

# ========================
# TTS with VTS Mouth Sync - natural smooth mouth movement
# ========================
mouth_moving = False

def natural_mouth_movement():
    global mouth_moving
    start_time = time.time()
    while mouth_moving:
        elapsed = (time.time() - start_time) % 2.0
        base_value = elapsed / 1.0 if elapsed < 1.0 else 2.0 - elapsed
        val = max(0.0, min(1.0, base_value + (random.random() - 0.5) * 0.15))
        try:
            ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "mouthFluctuate",
                "messageType": "InjectParameterDataRequest",
                "data": {
                    "parameterValues": [
                        {"id": "MouthOpen", "value": val}
                    ]
                }
            }))
        except Exception as e:
            print(f"[VTS] Error sending mouth value: {e}")
        time.sleep(0.05)
    try:
        ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "mouthClose",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {"id": "MouthOpen", "value": 0.0}
                ]
            }
        }))
    except Exception as e:
        print(f"[VTS] Error closing mouth: {e}")

def speak_text(text):
    global speaking, last_tts_end_time, mouth_moving
    speaking = True
    mouth_moving = True

    mouth_thread = threading.Thread(target=natural_mouth_movement)
    mouth_thread.start()

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    selected_voice = None
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            selected_voice = voice
            break
    if selected_voice:
        engine.setProperty('voice', selected_voice.id)

    engine.setProperty('outputDevice', 'CABLE Input (VB-Audio Virtual Cable)')

    engine.say(text)
    engine.runAndWait()

    mouth_moving = False
    mouth_thread.join()

    last_tts_end_time = time.time()
    speaking = False

# ========================
# Main Loop
# ========================
def main():
    connect_vts()
    authenticate_vts()
    print("Ready. Talk to your anime chatbot.")
    while True:
        user_input = recognize_speech()
        if user_input:
            ai_reply = get_ai_response(user_input)
            print(f"AI: {ai_reply}")
            speak_text(ai_reply)

if __name__ == "__main__":
    main()
