import speech_recognition as sr
import pyttsx3

from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Tsundere-style personality prompt
prompt_template = """
You are Amy, an anime girl who is a tsundere, someone who's not honest with their feelings. Please chat with me using this personality. 
All responses you give must be in first person.
Don't be overly mean, remember, you are not mean, just misunderstood. 
Do not ever break character. Do not admit you are a tsundere. 
Do not include any emojis or actions within the text that cannot be spoken. Do not explicitly say your name in your response. 

Current conversation:
{history}

Human: 
{input}
AI:
"""

# Prompt template setup
prompt_temp = PromptTemplate(template=prompt_template, input_variables=['history', 'input'])

# 🧠 Load Ollama LLM (make sure this model is installed in Ollama first)
llm = ChatOllama(
    model="llama3",  # Replace with 'mistral', 'gemma', etc. if desired
    temperature=0.8,
)

# Setup the conversation chain
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_temp,
    memory=ConversationBufferWindowMemory()
)

# 🎤 Recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for speech...")
        audio = recognizer.listen(source)

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

# 🧾 Get response from the AI
def get_openai_response(prompt):
    response = conversation.invoke({'input': str(prompt)})
    return str(response['response']).strip()

# 🔊 Speak the AI's response
def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Use a female voice
    engine.say(text)
    engine.runAndWait()

# 🚀 Main loop
def main():
    print("Welcome to the voice-activated chatbot!")
    while True:
        print("Say something:")
        user_input = recognize_speech()
        if user_input:
            response = get_openai_response(user_input)
            print(f"AI: {response}")
            speak_text(response)

if __name__ == "__main__":
    main()
