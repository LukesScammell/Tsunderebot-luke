import speech_recognition as sr
import pyttsx3
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from pythonosc.udp_client import SimpleUDPClient
import time
import os

print("Available microphone devices:")
for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {mic_name}")

# Set this to the index of your microphone from the list printed above
MIC_DEVICE_INDEX = 13  # Replace 1 with the correct index for your mic

# TTS/ASR coordination to avoid cutting off or listening during TTS
speaking = False
last_tts_end_time = 0.0
TTS_COOLDOWN_SECONDS = 0.9

# # ========= VSeeFace OSC Setup =========
# VSEEFACE_IP = "192.168.1.98"
# VSEEFACE_PORT = 39539
# osc = SimpleUDPClient(VSEEFACE_IP, VSEEFACE_PORT)

# osc.send_message("/VMC/Ext/Blend/Val", ["Joy", 1.0])
# time.sleep(2)
# osc.send_message("/VMC/Ext/Blend/Val", ["Joy", 0.0])

# def start_talking():
#     print("Sending /expression/joy to VSeeFace")
#     osc.send_message("/VMC/Ext/Blend/Val", ["Mouth_A", 100])  # mouth open
 
# def stop_talking():
#     print("Sending /expression/joy to VSeeFace")
#     osc.send_message("/VMC/Ext/Blend/Val", ["Mouth_A", 0])    # mouth closed

    

# def nod_head():
#     # VSeeFace head tracking - use bone rotation format
#     # Format: [x, y, z, w] quaternion for head bone
    
#     # Start with neutral position
#     osc.send_message("/VMC/Ext/Bone/Rot", ["Head", 0.0, 0.0, 0.0, 1.0])
#     time.sleep(0.1)
    
#     # Nod down (pitch rotation)
#     osc.send_message("/VMC/Ext/Bone/Rot", ["Head", -0.259, 0.0, 0.0, 0.966])  # ~30 degree nod
#     time.sleep(0.4)
    
#     # Back to neutral
#     osc.send_message("/VMC/Ext/Bone/Rot", ["Head", 0.0, 0.0, 0.0, 1.0])
#     time.sleep(0.2)
    
#     # Slight nod again
#     osc.send_message("/VMC/Ext/Bone/Rot", ["Head", -0.130, 0.0, 0.0, 0.991])  # ~15 degree nod
#     time.sleep(0.3)
    
#     # Return to neutral
#     osc.send_message("/VMC/Ext/Bone/Rot", ["Head", 0.0, 0.0, 0.0, 1.0])

# def trigger_expression(name, value=1.0):
#     osc.send_message("/VMC/Ext/Blend/Val", [name, float(value)])

# def clear_all_expressions():
#     expressions = ["Neutral", "Fun", "Angry", "Joy", "Sorrow", "Surprised"]
#     for exp in expressions:
#         trigger_expression(exp, 0.0)

# def detect_emotion_and_animate(text):
#     text_lower = text.lower()
#     clear_all_expressions()

#     if "baka" in text_lower or "idiot" in text_lower:
#         trigger_expression("Angry", 1.0)
#     elif "happy" in text_lower or "joy" in text_lower or "cute" in text_lower:
#         trigger_expression("Joy", 1.0)
#     elif "sad" in text_lower or "sorrow" in text_lower:
#         trigger_expression("Sorrow", 1.0)
#     elif "surprised" in text_lower or "wow" in text_lower:
#         trigger_expression("Surprised", 1.0)
#     elif "fun" in text_lower or "lol" in text_lower or "haha" in text_lower:
#         trigger_expression("Fun", 1.0)
#     else:
#         trigger_expression("Neutral", 1.0)

# ========= LangChain Setup =========
prompt_template = """
You are a sweet anime girl with a "Deretsun" personality — someone who is usually warm, caring, and affectionate, but occasionally gets flustered or teasing in a lighthearted way. Please chat with me using this personality.
All responses must be in first person.
You should speak gently and lovingly most of the time, but feel free to tease a little when the mood is right. 
Don't break character, don't say you're roleplaying or pretend you're fictional. Do not include any emojis or non-spoken actions. Do not explicitly say your name in your response.


Current conversation:
{history}

Human: 
{input}
AI:
"""

prompt_temp = PromptTemplate(template=prompt_template, input_variables=['history', 'input'])

llm = ChatOllama(model="llama3", temperature=0.8)

# ===== Persistent Memory (learn across runs) =====
# Store chat history as a JSON file in a local 'memory' folder next to this script
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
HISTORY_PATH = os.path.join(MEMORY_DIR, "chat_history.json")

history = FileChatMessageHistory(HISTORY_PATH)
memory = ConversationBufferMemory(
    chat_memory=history,
    memory_key="history",     # your PromptTemplate expects {history}
    return_messages=False       # keep history as a single string for your prompt
)

conversation = ConversationChain(
    llm=llm,
    prompt=prompt_temp,
    memory=memory
)

# ========= Speech Recognition =========
def recognize_speech():
    global speaking, last_tts_end_time, TTS_COOLDOWN_SECONDS
    recognizer = sr.Recognizer()
    # Make end-of-phrase detection less aggressive so you aren't cut off
    recognizer.pause_threshold = 1.5           # require a longer pause before stopping
    recognizer.non_speaking_duration = 0.7     # tolerate brief gaps while speaking
    recognizer.dynamic_energy_threshold = True # adapt initially, then freeze
    microphone = sr.Microphone(device_index=MIC_DEVICE_INDEX)  # Use explicit mic

    # If TTS just spoke, wait a short cooldown before listening
    while speaking:
        time.sleep(0.05)
    dt = time.time() - last_tts_end_time
    if 0 <= dt < TTS_COOLDOWN_SECONDS:
        time.sleep(TTS_COOLDOWN_SECONDS - dt)

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        # Freeze threshold so it doesn't drift during your speech
        recognizer.dynamic_energy_threshold = False
        print("Listening for speech...")
        # Allow longer sentences; increase timeout to wait for start of speech
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            print("No speech detected within timeout.")
            return None

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None
# ========= AI Response =========
def get_openai_response(prompt):
    response = conversation.invoke({'input': str(prompt)})
    return str(response['response']).strip()

# ========= TTS Output + Expression Sync =========
def speak_text(text):
    global speaking, last_tts_end_time
    speaking = True
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    selected_voice = None
    for voice in voices:
        if "female" in voice.name.lower() or "zira" in voice.name.lower():
            selected_voice = voice
            break

    if selected_voice:
        engine.setProperty('voice', selected_voice.id)
    else:
        engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)

    engine.setProperty('outputDevice', 'CABLE Input (VB-Audio Virtual Cable)')
    engine.say(text)
    engine.runAndWait()
    # Mark TTS finished and set a short cooldown before ASR listens again
    last_tts_end_time = time.time()
    speaking = False

# ========= Main Loop =========
def main():
    print("Welcome to the voice-activated tsundere chatbot with VSeeFace sync!")
    while True:
        print("Say something:")
        user_input = recognize_speech()
        if user_input:
            response = get_openai_response(user_input)
            print(f"AI: {response}")
            speak_text(response)

if __name__ == "__main__":
    main()
