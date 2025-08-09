import speech_recognition as sr

mic_list = sr.Microphone.list_microphone_names()
print("Available microphone devices:")
for i, mic_name in enumerate(mic_list):
    print(f"{i}: {mic_name}")