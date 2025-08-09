import speech_recognition as sr

print("Available microphones:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {name}")

mic_index = int(input("Enter mic index to test: "))

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=mic_index)

with mic as source:
    recognizer.adjust_for_ambient_noise(source)
    print("Say something:")
    audio = recognizer.listen(source)

try:
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")
except Exception as e:
    print(f"Error: {e}")