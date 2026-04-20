from gtts import gTTS

phrases = [
    "hello world",
    "how are you",
    "i am fine",
    "open chrome",
    "play music",
    "stop music",
    "what time is it",
    "close the window",
    "turn on wifi",
    "turn off wifi"
]

for i, text in enumerate(phrases):
    tts = gTTS(text)
    tts.save(f"audio_{i}.mp3")