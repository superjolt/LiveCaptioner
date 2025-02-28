import sys
import numpy as np
import sounddevice as sd
import whisper
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QInputDialog
from PyQt6.QtCore import Qt, QTimer

# Ask user for settings
def get_user_preferences():
    model_options = {
        "Fastest but less accurate": "tiny",
        "Balanced (medium speed & accuracy)": "small",
        "Best accuracy (slowest)": "large"
    }
    
    # Ask for model type
    model_choice, ok = QInputDialog.getItem(None, "Choose Model", 
                                            "Select the type of speech recognition model:", 
                                            list(model_options.keys()), 0, False)
    model_name = model_options[model_choice] if ok else "small"

    # Ask for update speed
    duration, ok = QInputDialog.getInt(None, "Update Speed", 
                                       "Enter how often captions should update (seconds):", 
                                       5, 1, 10, 1)
    duration = duration if ok else 5

    return model_name, duration

# Get user preferences
app = QApplication(sys.argv)
model_name, DURATION = get_user_preferences()

# Load the Whisper model
model = whisper.load_model(model_name)

# Set up audio parameters
SAMPLE_RATE = 16000  
BUFFER_SIZE = SAMPLE_RATE * DURATION

# GUI to display captions
class CaptionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(1200, 800, 400, 100)  # Position at bottom-right

        self.label = QLabel("Listening...", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; color: white; background-color: black; padding: 10px; border-radius: 10px;")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_transcription)
        self.timer.start(DURATION * 1000)

    def update_transcription(self):
        global audio_buffer
        audio_data = np.copy(audio_buffer)
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Transcribe
        result = model.transcribe(audio_data)
        text = result['text'] if result['text'] else "..."

        # Update label
        self.label.setText(text)

# Audio buffer to store data
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.int16)

# Audio callback function
def callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]  # Use first audio channel

# Start capturing system audio
with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, device=None):
    window = CaptionWindow()
    window.show()
    sys.exit(app.exec())
