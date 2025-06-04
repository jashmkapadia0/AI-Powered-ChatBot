import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import whisper
import google.generativeai as genai
from gtts import gTTS
import os
import pygame
import threading

# ---- CONFIG ----
API_KEY = "AIzaSyCTRrApB61GH3hBYZDbv0zs6GeF0dr1IAk"  # Replace this
DURATION = 5
SAMPLERATE = 16000
WIDTH, HEIGHT = 1366, 768

# ---- Pygame Setup ----
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice Q&A")
font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

def draw_text_lines(lines, y_start):
    for i, line in enumerate(lines):
        text_surf = font.render(line, True, (255, 255, 255))
        screen.blit(text_surf, (30, y_start + i * 40))

def render_wrapped_text(text, max_width):
    words = text.split(' ')
    lines = []
    line = ''
    for word in words:
        test_line = line + word + ' '
        if font.size(test_line)[0] < max_width:
            line = test_line
        else:
            lines.append(line.strip())
            line = word + ' '
    lines.append(line.strip())
    return lines

# ---- State ----
stage = "recording"
question_text = ""
answer_text = ""
answer_lines = []

def background_process():
    global stage, question_text, answer_text, answer_lines

    # --- Recording ---
    stage = "recording"
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        write(f.name, SAMPLERATE, recording)
        audio_path = f.name

    # --- Transcribe ---
    stage = "transcribing"
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    question_text = result['text']

    # --- Generate Answer ---
    stage = "answering"
    genai.configure(api_key=API_KEY)
    response = genai.GenerativeModel("models/gemini-1.5-flash-8b").generate_content(question_text)
    answer_text = response.text if hasattr(response, "text") else str(response)
    answer_lines = render_wrapped_text(answer_text, 740)

    # --- Convert only first 3 lines to speech ---
    speak_text = ' '.join(answer_lines[:3])
    tts = gTTS(text=speak_text, lang='en', slow=False)
    tts.save("output.mp3")

    # --- Play with Pygame ---
    stage = "speaking"
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)

    # --- Done ---
    stage = "done"

# Start background process
threading.Thread(target=background_process, daemon=True).start()

# --- Pygame Main Loop ---
running = True
while running:
    screen.fill((30, 30, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if stage == "recording":
        draw_text_lines(["🎙️ Recording your question..."], 250)
    elif stage == "transcribing":
        draw_text_lines(["🧠 Transcribing audio..."], 250)
    elif stage == "answering":
        draw_text_lines(["🤖 Getting an answer from Niru..."], 250)
    elif stage == "speaking":
        draw_text_lines(["🔊 Speaking "], 30)
        draw_text_lines(["A:"] + answer_lines, 250)
    elif stage == "done":
        draw_text_lines(["Q:"] + render_wrapped_text(question_text, 740), 30)
        draw_text_lines(["A:"] + answer_lines, 250)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
