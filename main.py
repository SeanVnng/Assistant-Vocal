import asyncio
import base64
import io
import os
import sys
import traceback
import json
import websockets
import argparse
import threading
import math
import subprocess
import webbrowser
from html import escape

# --- GUI Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLabel,
                               QVBoxLayout, QWidget, QLineEdit, QHBoxLayout,
                               QSizePolicy, QPushButton, QFrame, QScrollArea)
from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer
from PySide6.QtGui import (QImage, QPixmap, QPainter, QColor, QBrush, QPen, QTextCursor)

# --- Media & AI ---
import cv2
import pyaudio
import PIL.Image
from google import genai
from dotenv import load_dotenv
from PIL import ImageGrab
import numpy as np

# --- Config & Keys ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or not ELEVENLABS_API_KEY:
    sys.exit("Error: API Keys manquantes dans le fichier .env")

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_RATE = 16000
RECV_RATE = 24000
CHUNK = 1024
MODEL_ID = "gemini-live-2.5-flash-preview"
VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'

# ==============================================================================
# 1. ANIMATION AUDIO (Style Moderne)
# ==============================================================================
class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.bars = 20
        self.values = [0.1] * self.bars
        self.is_active = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def set_active(self, active):
        self.is_active = active
        if not active:
            self.values = [0.1] * self.bars
            self.update()

    def animate(self):
        if self.is_active:
            self.values = [max(0.1, min(1.0, v + (np.random.rand() - 0.5) * 0.5)) for v in self.values]
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_w = w / self.bars
        
        color = QColor(255, 170, 0) if self.is_active else QColor(80, 80, 80)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)

        for i, val in enumerate(self.values):
            bar_h = h * val * 0.8
            x = i * bar_w + 2
            y = (h - bar_h) / 2
            painter.drawRoundedRect(int(x), int(y), int(bar_w - 4), int(bar_h), 4, 4)

# ==============================================================================
# 2. BACKEND AI (Core Logic)
# ==============================================================================
class AssistantBackend(QObject):
    # Signals
    text_out = Signal(str)
    turn_complete = Signal()
    image_out = Signal(QImage)
    logs_out = Signal(str, str)
    mode_changed = Signal(str)
    voice_state = Signal(bool)

    def __init__(self, mode="none"):
        super().__init__()
        self.mode = mode
        self.running = True
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.pya = pyaudio.PyAudio()
        
        self.q_gemini_in = asyncio.Queue(maxsize=20)
        self.q_tts_in = asyncio.Queue()
        self.q_audio_out = asyncio.Queue()
        self.q_text_in = asyncio.Queue()

        self.latest_frame = None
        self.loop = asyncio.new_event_loop()
        
        self.tools_config = [
            {'google_search': {}}, {'code_execution': {}},
            {"function_declarations": [
                {"name": "create_folder", "description": "Create folder", "parameters": {"type": "OBJECT", "properties": {"folder_path": {"type": "STRING"}}, "required": ["folder_path"]}},
                {"name": "create_file", "description": "Create file", "parameters": {"type": "OBJECT", "properties": {"file_path": {"type": "STRING"}, "content": {"type": "STRING"}}, "required": ["file_path", "content"]}},
                {"name": "edit_file", "description": "Append to file", "parameters": {"type": "OBJECT", "properties": {"file_path": {"type": "STRING"}, "content": {"type": "STRING"}}, "required": ["file_path", "content"]}},
                {"name": "list_files", "description": "List dir", "parameters": {"type": "OBJECT", "properties": {"directory_path": {"type": "STRING"}}}},
                {"name": "read_file", "description": "Read file", "parameters": {"type": "OBJECT", "properties": {"file_path": {"type": "STRING"}}, "required": ["file_path"]}},
                {"name": "open_application", "description": "Open app", "parameters": {"type": "OBJECT", "properties": {"application_name": {"type": "STRING"}}, "required": ["application_name"]}},
                {"name": "open_website", "description": "Open URL", "parameters": {"type": "OBJECT", "properties": {"url": {"type": "STRING"}}, "required": ["url"]}},
            ]}
        ]

    def _run_tool(self, name, args):
        try:
            if name == "create_folder": os.makedirs(args["folder_path"], exist_ok=True); return "Folder created."
            if name == "create_file": 
                with open(args["file_path"], 'w') as f: f.write(args["content"])
                return "File created."
            if name == "edit_file":
                with open(args["file_path"], 'a') as f: f.write("\n" + args["content"])
                return "File updated."
            if name == "read_file":
                with open(args["file_path"], 'r') as f: return f.read()
            if name == "list_files":
                path = args.get("directory_path", ".")
                return str(os.listdir(path))
            if name == "open_website": webbrowser.open(args["url"]); return "Website opened."
            if name == "open_application":
                subprocess.Popen(args["application_name"], shell=True); return "App launched."
        except Exception as e: return f"Error: {e}"
        return "Unknown tool."

    async def task_camera(self):
        print(">>> [INIT] Module Vidéo démarré")
        cap = None
        
        while self.running:
            try:
                frame = None
                
                # --- MODE CAMERA (AUTO-DETECT) ---
                if self.mode == "camera":
                    # Si la caméra n'est pas ouverte, on la cherche
                    if cap is None or not cap.isOpened():
                        print(">>> [CAM] Recherche d'une caméra active...")
                        found = False
                        # On teste les ports 0, 1 et 2
                        for i in range(3):
                            temp_cap = await asyncio.to_thread(cv2.VideoCapture, i)
                            if temp_cap.isOpened():
                                ret, test_frame = await asyncio.to_thread(temp_cap.read)
                                if ret:
                                    print(f">>> [CAM] Caméra trouvée à l'index {i}")
                                    cap = temp_cap
                                    found = True
                                    break
                                else:
                                    temp_cap.release()
                        
                        if not found:
                            print(">>> [CAM] ERREUR: Aucune caméra fonctionnelle trouvée (testé 0, 1, 2).")
                            await asyncio.sleep(2) # On attend avant de retester
                            continue

                    if cap and cap.isOpened():
                        ret, frame_read = await asyncio.to_thread(cap.read)
                        if ret:
                            frame = frame_read
                        else:
                            print(">>> [CAM] Erreur lecture frame. Reset...")
                            cap.release()
                            cap = None
                
                # --- MODE ECRAN ---
                elif self.mode == "screen":
                    if cap:
                        await asyncio.to_thread(cap.release)
                        cap = None
                    
                    try:
                        screenshot = await asyncio.to_thread(ImageGrab.grab)
                        frame_np = np.array(screenshot)
                        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f">>> [SCREEN] Erreur capture écran: {e}")
                        await asyncio.sleep(1)

                # --- MODE OFF ---
                else:
                    if cap:
                        await asyncio.to_thread(cap.release)
                        cap = None
                    await asyncio.sleep(0.1)

                # --- ENVOI A L'INTERFACE ---
                if frame is not None:
                    self.latest_frame = frame
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    self.image_out.emit(qimg.copy())
                    await asyncio.sleep(0.03) 
                else:
                    self.image_out.emit(QImage())
                    
            except Exception as e:
                print(f">>> [VIDEO ERROR] {e}")
                await asyncio.sleep(1)

    async def task_gemini_vision_pusher(self):
        while self.running:
            await asyncio.sleep(1.0)
            if self.mode != "none" and self.latest_frame is not None:
                try:
                    rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                    pil = PIL.Image.fromarray(rgb)
                    pil.thumbnail((800, 800))
                    buf = io.BytesIO()
                    pil.save(buf, format="jpeg")
                    await self.q_gemini_in.put({"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()})
                except Exception:
                    pass

    async def task_audio_input(self):
        try:
            stream = self.pya.open(format=FORMAT, channels=CHANNELS, rate=SEND_RATE, input=True, frames_per_buffer=CHUNK)
            while self.running:
                data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
                await self.q_gemini_in.put({"data": data, "mime_type": "audio/pcm"})
        except Exception as e:
            print(f">>> [AUDIO INPUT ERROR] {e}")

    async def task_tts(self):
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_24000"
        while self.running:
            text = await self.q_tts_in.get()
            if not text: self.q_tts_in.task_done(); continue
            
            self.voice_state.emit(True)
            try:
                async with websockets.connect(uri) as ws:
                    await ws.send(json.dumps({"text": " ", "xi_api_key": ELEVENLABS_API_KEY}))
                    await ws.send(json.dumps({"text": text + " "}))
                    
                    while not self.q_tts_in.empty():
                        nxt = self.q_tts_in.get_nowait()
                        if nxt: await ws.send(json.dumps({"text": nxt + " "}))
                        self.q_tts_in.task_done()
                    await ws.send(json.dumps({"text": ""}))

                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("audio"): await self.q_audio_out.put(base64.b64decode(data["audio"]))
                        if data.get("isFinal"): break
            except Exception as e: print(f"TTS Error: {e}")
            finally: 
                self.voice_state.emit(False)
                self.q_tts_in.task_done()

    async def task_audio_output(self):
        try:
            stream = self.pya.open(format=pyaudio.paInt16, channels=1, rate=RECV_RATE, output=True)
            while self.running:
                data = await self.q_audio_out.get()
                await asyncio.to_thread(stream.write, data)
                self.q_audio_out.task_done()
        except Exception as e:
            print(f">>> [AUDIO OUTPUT ERROR] {e}")

    async def run_core(self):
        # --- INSTRUCTION SYSTEME CORRIGÉE POUR LA VISION ---
        sys_instruction = """
        You are NEXUS AI, an advanced assistant with REAL-TIME VISION.
        IMPORTANT: You are receiving a continuous video stream.
        1. If the user is sharing their screen (Screen Mode) or Camera, YOU CAN SEE IT.
        2. Never say "I cannot see" or "I am text-based". Use the visual data provided.
        3. If the image is black/empty, ask the user to check their camera/screen share.
        4. Be concise, modern, and helpful.
        """
        
        config = {
            "response_modalities": ["TEXT"], 
            "tools": self.tools_config, 
            "system_instruction": sys_instruction
        }
        
        try:
            async with self.client.aio.live.connect(model=MODEL_ID, config=config) as session:
                self.session = session
                
                asyncio.create_task(self.task_camera())
                asyncio.create_task(self.task_gemini_vision_pusher())
                asyncio.create_task(self.task_audio_input())
                asyncio.create_task(self.task_tts())
                asyncio.create_task(self.task_audio_output())

                async def sender():
                    while self.running:
                        item = await self.q_gemini_in.get()
                        await session.send(input=item)
                        self.q_gemini_in.task_done()
                asyncio.create_task(sender())

                async def text_input_handler():
                    while self.running:
                        txt = await self.q_text_in.get()
                        if txt: await session.send(input=txt, end_of_turn=True)
                        self.q_text_in.task_done()
                asyncio.create_task(text_input_handler())

                while self.running:
                    async for chunk in session.receive():
                        if chunk.tool_call:
                            responses = []
                            for fc in chunk.tool_call.function_calls:
                                res = self._run_tool(fc.name, fc.args)
                                responses.append({"id": fc.id, "name": fc.name, "response": {"result": res}})
                                self.logs_out.emit("TOOL", f"{fc.name}: {str(fc.args)} -> {res}")
                            await session.send_tool_response(function_responses=responses)
                        
                        if chunk.server_content:
                            if chunk.server_content.model_turn:
                                 for p in chunk.server_content.model_turn.parts:
                                    if p.executable_code: 
                                        self.logs_out.emit("CODE", p.executable_code.code)
                                    if p.code_execution_result:
                                        self.logs_out.emit("OUTPUT", p.code_execution_result.output)

                            txt = chunk.text
                            if txt:
                                self.text_out.emit(txt)
                                await self.q_tts_in.put(txt)
                    
                    self.turn_complete.emit()
                    await self.q_tts_in.put(None)
        except Exception as e:
            print(f">>> CRITICAL ERROR: {e}")

    def start(self):
        threading.Thread(target=lambda: self.loop.run_until_complete(self.run_core()), daemon=True).start()

    def stop(self): self.running = False
    
    @Slot(str)
    def set_mode(self, mode):
        self.mode = mode; self.mode_changed.emit(mode)
    
    @Slot(str)
    def user_input(self, text):
        self.loop.call_soon_threadsafe(self.q_text_in.put_nowait, text)

# ==============================================================================
# 3. MODERN GUI
# ==============================================================================
class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXUS AI")
        self.resize(1300, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #f0f0f0; font-family: 'Segoe UI', sans-serif; }
            QWidget { font-size: 14px; }
            QFrame#panel { background-color: #252526; border-radius: 12px; border: 1px solid #333; }
            QLabel#log_title { color: #ffb300; font-weight: bold; font-size: 12px; letter-spacing: 1px; }
            QTextEdit#logs { background: transparent; border: none; color: #aaaaaa; font-family: 'Consolas', monospace; font-size: 12px; }
            QTextEdit#chat { background: transparent; border: none; padding: 10px; font-size: 15px; line-height: 1.4; }
            QLineEdit { 
                background-color: #2d2d30; border: 2px solid #3e3e42; border-radius: 20px; 
                padding: 10px 20px; color: white; selection-background-color: #ffb300;
            }
            QLineEdit:focus { border: 2px solid #ffb300; }
            QPushButton {
                background-color: #2d2d30; color: #dddddd; border: none; border-radius: 8px;
                padding: 8px 16px; font-weight: 600;
            }
            QPushButton:hover { background-color: #3e3e42; color: white; }
            QPushButton:checked { background-color: #ffb300; color: #1e1e1e; }
            QScrollBar:vertical { background: #1e1e1e; width: 8px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setContentsMargins(20, 20, 20, 20); main_layout.setSpacing(20)

        left_col = QVBoxLayout()
        self.chat_display = QTextEdit(); self.chat_display.setObjectName("chat")
        self.chat_display.setReadOnly(True)
        chat_frame = QFrame(); chat_frame.setObjectName("panel"); chat_layout = QVBoxLayout(chat_frame)
        chat_layout.addWidget(self.chat_display)
        left_col.addWidget(chat_frame, 1)

        self.visualizer = AudioVisualizer()
        left_col.addWidget(self.visualizer)

        inp_layout = QHBoxLayout()
        self.input_box = QLineEdit(); self.input_box.setPlaceholderText("Message Nexus AI...")
        self.input_box.returnPressed.connect(self.send_message)
        inp_layout.addWidget(self.input_box)
        left_col.addLayout(inp_layout)
        main_layout.addLayout(left_col, 6)

        right_col = QVBoxLayout()
        self.video_label = QLabel("Camera Off"); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 12px; color: #555;")
        self.video_label.setMinimumHeight(250)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.video_label)

        ctrl_layout = QHBoxLayout()
        self.btn_cam = QPushButton("CAMERA"); self.btn_cam.setCheckable(True)
        self.btn_scr = QPushButton("SCREEN"); self.btn_scr.setCheckable(True)
        self.btn_off = QPushButton("OFF"); self.btn_off.setCheckable(True); self.btn_off.setChecked(True)
        
        self.btn_cam.clicked.connect(lambda: self.switch_mode("camera"))
        self.btn_scr.clicked.connect(lambda: self.switch_mode("screen"))
        self.btn_off.clicked.connect(lambda: self.switch_mode("none"))
        
        ctrl_layout.addWidget(self.btn_cam); ctrl_layout.addWidget(self.btn_scr); ctrl_layout.addWidget(self.btn_off)
        right_col.addLayout(ctrl_layout)

        log_frame = QFrame(); log_frame.setObjectName("panel")
        log_layout = QVBoxLayout(log_frame)
        log_layout.addWidget(QLabel("SYSTEM LOGS", objectName="log_title"))
        self.logs_display = QTextEdit(); self.logs_display.setObjectName("logs"); self.logs_display.setReadOnly(True)
        log_layout.addWidget(self.logs_display)
        right_col.addWidget(log_frame, 1)

        main_layout.addLayout(right_col, 4)

        self.backend = AssistantBackend()
        self.backend.text_out.connect(self.append_text)
        self.backend.image_out.connect(self.update_video)
        self.backend.logs_out.connect(self.append_log)
        self.backend.turn_complete.connect(lambda: self.chat_display.append(""))
        self.backend.voice_state.connect(self.visualizer.set_active)
        self.backend.start()

    def send_message(self):
        text = self.input_box.text().strip()
        if text:
            self.chat_display.append(f"<div style='color:#aaaaaa; margin-top:10px;'>YOU: {escape(text)}</div>")
            self.backend.user_input(text)
            self.input_box.clear()

    @Slot(str)
    def append_text(self, text):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(f"<span style='color:#ffffff;'>{escape(text)}</span>")
        self.chat_display.setTextCursor(cursor)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    @Slot(str, str)
    def append_log(self, title, content):
        self.logs_display.append(f"<b style='color:#ffb300;'>[{title}]</b> {escape(content)}")
    
    @Slot(QImage)
    def update_video(self, image):
        if image.isNull(): 
            self.video_label.clear(); self.video_label.setText("Camera Off")
        else:
            pix = QPixmap.fromImage(image)
            self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def switch_mode(self, mode):
        self.btn_cam.setChecked(mode == "camera")
        self.btn_scr.setChecked(mode == "screen")
        self.btn_off.setChecked(mode == "none")
        self.backend.set_mode(mode)

    def closeEvent(self, event):
        self.backend.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())