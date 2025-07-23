import sys
import threading
import queue
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from optimum.bettertransformer import BetterTransformer

os.environ["HF_TOKEN"] = "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"

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer
)

from PyQt6.QtGui import QTextCursor, QFont

from ultravox.inference.ultravox_infer import UltravoxInference
from ultravox import data as datasets
from ultravox.inference import base

MODEL_PATH = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_TYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

# WHISPER_MODEL_ID = "openai/whisper-tiny"
WHISPER_MODEL_ID = "openai/whisper-medium"

WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_DTYPE = torch.float16 if WHISPER_DEVICE == "cuda" else torch.float32

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
AUDIO_BUFFER_SECONDS = 3
VAD_THRESHOLD = 0.01 
SILENCE_DETECTION_DURATION = 1.0

AUDIO_SAVE_DIR = "user_audio_segments"
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

audio_q = queue.Queue()
llm_input_q = queue.Queue()
stop_event = threading.Event()
interrupt_event = threading.Event()

is_recording_active = False

class LLMWorker(QThread):
    status_updated = pyqtSignal(str)
    call_buttons_enabled = pyqtSignal()
    call_buttons_disabled = pyqtSignal()
    llm_chunk_received = pyqtSignal(str)
    llm_error_received = pyqtSignal(str)
    llm_finished = pyqtSignal()

    def __init__(self, model_path, device, data_type):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.data_type = data_type
        self.inference_pipeline: Optional[UltravoxInference] = None
        self.current_llm_response = "" # Moved from PhoneCallApp to here

    def run(self):
        try:
            self.status_updated.emit("Loading AI model... This may take a moment.")
            self.inference_pipeline = UltravoxInference(
                model_path=self.model_path,
                device=self.device,
                data_type=self.data_type,
                conversation_mode=True,
            )
            self.status_updated.emit("AI model loaded. Ready for conversation.")
            self.call_buttons_enabled.emit()

            while not stop_event.is_set():
                try:
                    input_data = llm_input_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if input_data is None:
                    break

                audio_data = input_data.get('audio')
                sampling_rate = input_data.get('sampling_rate')
                current_messages = input_data.get('messages')
                
                full_conversation_reset = input_data.get('full_conversation_reset', False)

                if full_conversation_reset and self.inference_pipeline:
                    self.inference_pipeline.update_conversation(
                        past_messages=current_messages, 
                        past_key_values=None
                    )
                    if audio_data is None and (not current_messages or len(current_messages) <= 1): 
                        continue

                # Remove any previous <|audio|> placeholders for the current turn if they are not the only thing
                # This logic assumes the *last* user message might contain <|audio|> if actual audio is provided.
                messages_for_sample = []
                for i, msg in enumerate(current_messages):
                    if msg.get("role") == "user" and msg.get("content") == "<|audio|>" and audio_data is None:
                        # If audio_data is None, and we encounter <|audio|>, it means it was a placeholder for
                        # a previous attempt, so we might want to clean it up or ensure it's handled.
                        # For now, let's ensure only one <|audio|> is used when audio_data is present.
                        pass
                    else:
                        messages_for_sample.append(msg)
                
                # Special handling for when audio_data is present: ensure <|audio|> is the *last* content of the *last* user message
                if audio_data is not None:
                    if not messages_for_sample or messages_for_sample[-1].get("role") != "user":
                        messages_for_sample.append({"role": "user", "content": "<|audio|>"})
                    elif "<|audio|>" not in messages_for_sample[-1]["content"]:
                        messages_for_sample[-1]["content"] += " <|audio|>"
                    # If it's already there, do nothing.

                if audio_data is not None:
                    sample = datasets.VoiceSample(
                        messages=messages_for_sample, 
                        audio=audio_data,
                        sample_rate=sampling_rate
                    )
                else:
                    sample = datasets.VoiceSample(messages=messages_for_sample)

                self.status_updated.emit("AI is thinking...")
                self.current_llm_response = ""
                try:
                    stream_generator = self.inference_pipeline.infer_stream(sample)

                    for chunk in stream_generator:
                        if interrupt_event.is_set():
                            break 
                        if isinstance(chunk, base.InferenceChunk):
                            text_chunk = chunk.text
                            if text_chunk:
                                self.current_llm_response += text_chunk
                                self.llm_chunk_received.emit(text_chunk)
                        elif isinstance(chunk, base.InferenceStats):
                            if hasattr(chunk, "input_token_len") and hasattr(chunk, "output_token_len"):
                                print(f"LLM Inference Stats: Input Tokens: {chunk.input_token_len}, Output Tokens: {chunk.output_token_len}")
                            else:
                                print("LLM Inference Stats: (Token info unavailable)")

                    if not interrupt_event.is_set():
                        self.llm_finished.emit() 
                    else:
                        self.inference_pipeline.update_conversation(
                            past_messages=messages_for_sample, # Keep current messages as base
                            past_key_values=None 
                        )

                except Exception as e:
                    error_msg = f"LLM Inference Error: {e}"
                    self.status_updated.emit(f"AI Error: {e}")
                    self.llm_error_received.emit(error_msg)
                    if self.inference_pipeline:
                        self.inference_pipeline.update_conversation(
                            past_messages=[], # Reset completely on LLM error
                            past_key_values=None 
                        )
                finally:
                    interrupt_event.clear()

        except Exception as e:
            error_msg = f"LLM Thread Initialization Error: {e}"
            self.status_updated.emit(f"Fatal AI Error: {e}. Check console.")
            self.call_buttons_disabled.emit()
        finally:
            print("LLM Worker Thread Exiting.")

class PhoneCallApp(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ultravox Phone Call Demo (PyQt6)")
        self.setGeometry(100, 100, 600, 500)

        self.llm_worker_thread: Optional[LLMWorker] = None
        self.audio_stream: Optional[sd.InputStream] = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_start_time: Optional[float] = None
        self.is_speaking = False
        self.current_llm_response = ""
        self.chat_history_messages: List[Dict[str, str]] = []

        self.llm_cursor_position = None
        
        self._init_ui() 

        self.whisper_processor = None
        self.whisper_model = None
        self._load_whisper_model() 

        self._init_threads()

        self.audio_process_timer = QTimer(self)
        self.audio_process_timer.timeout.connect(self.process_audio_chunks)
        self.audio_process_timer.start(50)

    def _load_whisper_model(self):
        self.update_status("Loading Whisper ASR model...") 
        try:
            self.whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID,
                torch_dtype=WHISPER_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(WHISPER_DEVICE)

            if WHISPER_DEVICE == "cuda" and torch.cuda.is_available():
                try:
                    self.whisper_model = BetterTransformer.transform(self.whisper_model, keep_original_model=False)
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Failed to apply BetterTransformer to Whisper: {e}")

            self.update_status("Whisper ASR model loaded.")
        except Exception as e:
            self.update_status(f"Error loading Whisper ASR model: {e}")
            QMessageBox.critical(self, "Whisper ASR Error", f"Could not load Whisper ASR model: {e}\n\nMake sure you have `transformers` and `optimum` (if using BetterTransformer) installed and a compatible GPU/PyTorch setup.")
            self.whisper_model = None

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        main_layout.addWidget(self.status_label)

        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        font_conv = QFont()
        font_conv.setPointSize(10)
        self.conversation_display.setFont(font_conv)
        main_layout.addWidget(self.conversation_display)

        input_frame = QHBoxLayout()
        self.user_input_entry = QLineEdit()
        font_input = QFont()
        font_input.setPointSize(10)
        self.user_input_entry.setFont(font_input)
        self.user_input_entry.returnPressed.connect(self.send_text_message)
        input_frame.addWidget(self.user_input_entry)

        self.send_button = QPushButton("Send Text")
        self.send_button.clicked.connect(self.send_text_message)
        input_frame.addWidget(self.send_button)
        main_layout.addLayout(input_frame)

        control_frame = QHBoxLayout()
        self.start_call_button = QPushButton("Start Call")
        self.start_call_button.clicked.connect(self.start_call)
        self.start_call_button.setEnabled(False)
        control_frame.addWidget(self.start_call_button)

        self.end_call_button = QPushButton("End Call")
        self.end_call_button.clicked.connect(self.end_call)
        self.end_call_button.setEnabled(False)
        control_frame.addWidget(self.end_call_button)
        main_layout.addLayout(control_frame)

    def _init_threads(self):
        self.llm_worker_thread = LLMWorker(MODEL_PATH, DEVICE, DATA_TYPE)
        self.llm_worker_thread.status_updated.connect(self.update_status)
        self.llm_worker_thread.call_buttons_enabled.connect(self.enable_call_buttons)
        self.llm_worker_thread.call_buttons_disabled.connect(self.disable_call_buttons)
        self.llm_worker_thread.llm_chunk_received.connect(self.handle_llm_chunk)
        self.llm_worker_thread.llm_error_received.connect(self.handle_llm_error)
        self.llm_worker_thread.llm_finished.connect(self.handle_llm_finished)
        self.llm_worker_thread.start()

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(message)

    @pyqtSlot()
    def enable_call_buttons(self):
        self.start_call_button.setEnabled(True)
        self.end_call_button.setEnabled(False)

    @pyqtSlot()
    def disable_call_buttons(self):
        self.start_call_button.setEnabled(False)
        self.end_call_button.setEnabled(False)

    def append_to_conversation(self, speaker, text, color="black"):
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(f"<b style='color:{color};'>{speaker}:</b> {text}<br>")
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()

    @pyqtSlot(str)
    def handle_llm_chunk(self, chunk):
        self.current_llm_response += chunk
        text_to_display = self.current_llm_response

        if self.llm_cursor_position is None:
            cursor = self.conversation_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.llm_cursor_position = cursor.position()  
            cursor.insertHtml(f"<b style='color:purple;'>AI:</b> {text_to_display}")
            self.conversation_display.setTextCursor(cursor)
        else:
            cursor = self.conversation_display.textCursor()
            cursor.setPosition(self.llm_cursor_position)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(f"<b style='color:purple;'>AI:</b> {text_to_display}")
            self.conversation_display.setTextCursor(cursor)

        self.conversation_display.ensureCursorVisible()

    @pyqtSlot(str)
    def handle_llm_error(self, error_msg):
        self.append_to_conversation("AI", f"[ERROR]: {error_msg}", color="red")
        self.update_status("Error from AI. Resetting conversation. Please try again.")
        self.current_llm_response = ""
        self.llm_cursor_position = None
        self.user_input_entry.setEnabled(True)
        self.send_button.setEnabled(True)
        
        # Reset chat history and inform LLM worker to reset its internal state
        self.chat_history_messages = [{"role": "system", "content": "You are a helpful AI assistant. Respond concisely in a conversational manner."}]
        llm_input_q.put({"messages": self.chat_history_messages.copy(), "audio": None, "full_conversation_reset": True})

        # Clear audio buffer and reset flags
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_start_time = None
        self.is_speaking = False
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

    @pyqtSlot()
    def handle_llm_finished(self):
        if self.current_llm_response and not interrupt_event.is_set():
            self.chat_history_messages.append({"role": "assistant", "content": self.current_llm_response})
            cursor = self.conversation_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml("<br>")
            self.conversation_display.setTextCursor(cursor)
        elif interrupt_event.is_set():
            self.append_to_conversation("System", "AI generation interrupted.", color="orange")
        
        self.update_status("Listening...")
        self.llm_cursor_position = None
        self.current_llm_response = ""
        self.user_input_entry.setEnabled(True)
        self.send_button.setEnabled(True)

    def start_call(self):
        global is_recording_active
        if is_recording_active:
            self.update_status("Call already active.")
            return

        self.update_status("Starting call...")
        self.start_call_button.setEnabled(False)
        self.end_call_button.setEnabled(True)
        is_recording_active = True
        
        self.chat_history_messages = [{"role": "system", "content": "You are a helpful AI assistant. Respond concisely in a conversational manner."}]
        self.conversation_display.clear()
        self.append_to_conversation("System", "Call started. Say something!", color="blue")
        
        if self.llm_worker_thread: 
            llm_input_q.put({"messages": self.chat_history_messages.copy(), "audio": None, "full_conversation_reset": True})

        try:
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=CHANNELS, 
                dtype='float32',
                callback=self.audio_callback
            )
            self.audio_stream.start()
            self.update_status("Listening...")
        except Exception as e:
            self.update_status(f"Error starting audio stream: {e}")
            QMessageBox.critical(self, "Audio Error", f"Could not start audio stream: {e}\n\nMake sure your microphone is connected and not in use by another application.")
            self.end_call()

    def end_call(self):
        global is_recording_active
        if not is_recording_active:
            self.update_status("Call not active.")
            return

        self.update_status("Ending call...")
        is_recording_active = False
        self.start_call_button.setEnabled(True)
        self.end_call_button.setEnabled(False)

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_start_time = None
        self.is_speaking = False
        self.update_status("Call ended.")
        self.append_to_conversation("System", "Call ended.", color="blue")
        
        # Reset history and signal LLM worker to reset its state
        self.chat_history_messages = [] 
        if self.llm_worker_thread: 
            llm_input_q.put({"messages": [], "audio": None, "full_conversation_reset": True})
        
        while not llm_input_q.empty():
            try:
                llm_input_q.get_nowait()
            except queue.Empty:
                break

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Sounddevice warning: {status}")
        if not is_recording_active:
            return
        audio_chunk = indata[:, 0]
        audio_q.put(audio_chunk)

    @pyqtSlot()
    def process_audio_chunks(self):
        if not is_recording_active:
            return

        try:
            while not audio_q.empty():
                chunk = audio_q.get_nowait()
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

                volume = np.sqrt(np.mean(chunk**2))
                is_current_speech = volume > VAD_THRESHOLD

                if is_current_speech:
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.silence_start_time = None
                        self.update_status("User speaking...")
                        if self.current_llm_response: 
                            interrupt_event.set()
                            self.current_llm_response = ""
                            self.llm_cursor_position = None
                            # Clear LLM input queue on interruption
                            while not llm_input_q.empty():
                                try:
                                    llm_input_q.get_nowait()
                                except queue.Empty:
                                    break
                elif self.is_speaking:
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                    elif (time.time() - self.silence_start_time) > SILENCE_DETECTION_DURATION:
                        self.is_speaking = False
                        self.update_status("User finished speaking. Processing...")
                        self.process_user_turn()
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_start_time = None

                max_samples = int(AUDIO_BUFFER_SECONDS * SAMPLE_RATE)
                if len(self.audio_buffer) > max_samples:
                    self.audio_buffer = self.audio_buffer[-max_samples:]

        except Exception as e:
            print(f"Error in process_audio_chunks: {e}")
            self.update_status(f"Audio processing error: {e}")

    def process_user_turn(self, text_input: Optional[str] = None):
        self.current_llm_response = ""
        self.llm_cursor_position = None

        if text_input:
            user_message = {"role": "user", "content": text_input}
            self.append_to_conversation("You (Text)", text_input, color="darkgreen")
            self.chat_history_messages.append(user_message)
            
            # For text input, no audio data is sent.
            llm_input_q.put({"messages": self.chat_history_messages.copy(), "audio": None})

        elif len(self.audio_buffer) > 0:
            audio_for_llm = self.audio_buffer.copy()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(AUDIO_SAVE_DIR, f"user_audio_{timestamp}.wav")
            try:
                sf.write(filename, audio_for_llm, SAMPLE_RATE)
                self.append_to_conversation("System", f"Saved audio segment.", color="gray")
            except Exception as e:
                self.append_to_conversation("System", f"Error saving audio: {e}", color="red")

            transcribed_text = ""
            asr_success = False
            if self.whisper_model and self.whisper_processor:
                try:
                    # Ensure audio is 1D
                    audio_tensor = torch.tensor(audio_for_llm, dtype=torch.float32)
                    if audio_tensor.ndim > 1:
                        audio_tensor = audio_tensor.squeeze()
                    if len(audio_tensor) == 0:
                        raise ValueError("Empty audio tensor.")

                    input_features = self.whisper_processor(
                        audio_tensor.numpy(),  # Whisper expects NumPy input here
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt"
                    ).input_features.to(WHISPER_DTYPE).to(WHISPER_DEVICE)

                    predicted_ids = self.whisper_model.generate(input_features)
                    transcribed_text = self.whisper_processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]
                    self.append_to_conversation("You (ASR)", f"\"{transcribed_text}\"", color="darkblue")
                    asr_success = True
                except Exception as e:
                    print(f"Error during ASR transcription: {e}")
                    self.append_to_conversation("You (ASR)", f"[Transcription Error: {e}]", color="red")
                    transcribed_text = "[Audio input received, but transcription failed]"

            else:
                self.append_to_conversation("You (ASR)", "[Whisper model not loaded]", color="red")
                transcribed_text = "[Audio input received, Whisper not available]" # Descriptive placeholder

            self.append_to_conversation("You (Voice)", "[Audio Input Sent]", color="darkgreen") 

            messages_for_llm_worker = self.chat_history_messages.copy()
            
            # Append the user message with the transcribed text (or placeholder)
            # and explicitly add the <|audio|> token if actual audio is being sent.
            user_content = transcribed_text
            if asr_success:
                 # If ASR was successful, append the transcription
                messages_for_llm_worker.append({"role": "user", "content": user_content + " <|audio|>"})
            else:
                # If ASR failed, use the placeholder but still include <|audio|> for the model
                # assuming the model still expects audio when the `audio` field is populated.
                messages_for_llm_worker.append({"role": "user", "content": user_content + " <|audio|>"})
            
            llm_input_q.put({
                "audio": audio_for_llm,
                "sampling_rate": SAMPLE_RATE,
                "messages": messages_for_llm_worker
            })
            
            # Update chat history displayed in the GUI with the (potentially failed) transcription
            self.chat_history_messages.append({"role": "user", "content": transcribed_text})
        else:
            self.update_status("No audio or text input detected.")
            return

        self.user_input_entry.setEnabled(False)
        self.send_button.setEnabled(False)

    def send_text_message(self):
        message = self.user_input_entry.text().strip()
        if message:
            self.user_input_entry.clear()
            self.process_user_turn(text_input=message)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            stop_event.set()
            llm_input_q.put(None)
            
            if self.llm_worker_thread and self.llm_worker_thread.isRunning():
                self.llm_worker_thread.wait(5000)

            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()

            self.audio_process_timer.stop()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhoneCallApp()
    window.show()
    sys.exit(app.exec())