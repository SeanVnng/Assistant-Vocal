import sys
import os 
import io
import base64
import asyncio
import threading
import subprocess 
import webbrowser
from html import escape
import time

# --- IMPORTS AUDIO ---
import speech_recognition as sr
import pyttsx3 
from dotenv import load_dotenv 

# --- IMPORT GROQ ---
from groq import Groq

# --- GUI Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLabel,
    QVBoxLayout, QWidget, QLineEdit, QHBoxLayout,
    QPushButton, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, Slot, Signal, QObject, QTimer 
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QPainter, QColor, QBrush 

import cv2
import numpy as np
from PIL import ImageGrab, Image as PILImage

# --- CONFIGURATION ---
# Charge les variables du fichier .env
load_dotenv()

# Récupération des variables d'environnement
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_TEXT = os.getenv("MODEL_TEXT", "meta-llama/llama-4-scout-17b-16e-instruct")
MODEL_VISION = os.getenv("MODEL_VISION", "meta-llama/llama-4-scout-17b-16e-instruct")

# Vérification de sécurité
if not GROQ_API_KEY:
    print("ERREUR CRITIQUE : La clé GROQ_API_KEY est introuvable.")
    print("   Assurez-vous d'avoir créé le fichier .env avec votre clé.")
    sys.exit(1)

MODE_TEXT = 0
MODE_CAMERA = 1
MODE_SCREEN = 2

# ======================================================
# 1. VISUALIZER (Style Orange - D.A. main.py)
# ======================================================
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

# ======================================================
# 2. BACKEND (Logique main2.py - Groq & System)
# ======================================================
class Backend(QObject):
    text_out = Signal(str)
    image_out = Signal(QImage)
    logs_out = Signal(str, str)
    voice_state = Signal(bool) 
    stt_status = Signal(str)

    def __init__(self):
        super().__init__()
        self.mode = MODE_TEXT
        self.running = True
        self.current_frame = None  
        self.stt_active = False 

        self.loop = asyncio.new_event_loop()
        
        # Initialisation GROQ avec la clé sécurisée
        try:
            self.client = Groq(api_key=GROQ_API_KEY)
            print("Groq Connecté via .env")
        except Exception as e:
            print(f"ERREUR Connexion Groq: {e}")
            self.client = None

        self.q_text = asyncio.Queue()
        self.q_vision = asyncio.Queue()  
        
        self.recognizer = sr.Recognizer()

    def start(self):
        threading.Thread(
            target=lambda: self.loop.run_until_complete(self.main_loop()),
            daemon=True
        ).start()

    def stop(self):
        self.running = False
        
    # --- TTS (Text to Speech) ---
    def speak_sync(self, text):
        try:
            self.voice_state.emit(True)
            clean_text = text.replace("*", "").replace("#", "")
            if "<EXECUTE>" in clean_text:
                clean_text = clean_text.split("<EXECUTE>")[0] 
            
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if "fr" in voice.id.lower() or "french" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 170)
            engine.say(clean_text)
            engine.runAndWait() 
            
        except Exception as e:
            self.logs_out.emit("ERR", f"Erreur Audio: {e}")
        finally:
            self.voice_state.emit(False)

    async def speak(self, text):
        await asyncio.to_thread(self.speak_sync, text)

    # --- STT (Speech to Text) ---
    async def task_stt_worker(self):
        self.stt_status.emit("Prêt")
        while self.running:
            if not self.stt_active:
                await asyncio.sleep(0.2)
                continue
            
            self.stt_status.emit("Écoute...")
            self.logs_out.emit("INFO", "Micro ouvert...")

            text = await asyncio.to_thread(self.listen_sync)

            if text:
                self.logs_out.emit("INFO", f"Entendu: {text}")
                if self.mode == MODE_TEXT:
                    self.loop.call_soon_threadsafe(self.q_text.put_nowait, text)
                else:
                    self.loop.call_soon_threadsafe(self.q_vision.put_nowait, text)

                self.stt_active = False 
                self.stt_status.emit("Traitement...")
            else:
                self.stt_active = False
                self.stt_status.emit("Rien entendu")

    def listen_sync(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return self.recognizer.recognize_google(audio, language="fr-FR")
        except:
            return None

    # --- BOUCLE PRINCIPALE ---
    async def main_loop(self):
        self.logs_out.emit("SYSTEM", f"NEXUS Démarré sur GROQ 🚀")
        
        asyncio.create_task(self.camera_loop())
        asyncio.create_task(self.screen_loop())
        asyncio.create_task(self.task_stt_worker())

        SYSTEM_PROMPT = (
            "Tu es NEXUS, un assistant virtuel."
            "RÈGLES STRICTES :"
            "1. Réponds UNIQUEMENT en Français."
            "2. Fais des phrases TRÈS COURTES."
            "3. Si l'utilisateur veut ouvrir un site/appli, utilise UNIQUEMENT le format:"
            "   <EXECUTE>commande</EXECUTE>"
            "   Exemples: <EXECUTE>https://google.com</EXECUTE>, <EXECUTE>spotify</EXECUTE>"
        )

        while self.running:
            # Traitement Texte
            if not self.q_text.empty():
                user_text = await self.q_text.get()
                self.text_out.emit(user_text) 
                
                response = await self.model_call_text(user_text, SYSTEM_PROMPT)
                
                if "<EXECUTE>" in response:
                    parts = response.split("<EXECUTE>")
                    spoken_part = parts[0]
                    cmd = parts[1].split("</EXECUTE>")[0]
                    self.execute_system_command(cmd)
                    response = spoken_part + f" (Action: {cmd})"
                
                self.text_out.emit(f"NEXUS: {response}") 
                await self.speak(response)
                self.q_text.task_done()
            
            # Traitement Vision
            if not self.q_vision.empty():
                user_prompt = await self.q_vision.get()
                self.text_out.emit(f"[VISION] {user_prompt}")
                await self.analyze_image_groq(user_prompt)
                self.q_vision.task_done()

            await asyncio.sleep(0.05)

    # --- API CALLS ---
    async def model_call_text(self, user_input, sys_prompt):
        if not self.client: return "Erreur API."
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=MODEL_TEXT,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7, max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e: return f"Erreur: {str(e)}"

    async def analyze_image_groq(self, prompt):
        if self.current_frame is None or not self.client: 
            self.logs_out.emit("WARN", "Vision impossible : Pas d'image")
            return
        
        try:
            self.logs_out.emit("VISION", "Analyse en cours...")
            frame = self.current_frame
            scale = 800 / frame.shape[1]
            dim = (800, int(frame.shape[0] * scale))
            frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            
            _, buffer = cv2.imencode('.jpg', frame_resized)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            vision_instructions = (
                "Réponds en Français, sois bref. "
                "Si je demande d'ouvrir une appli ou un site, réponds UNIQUEMENT: "
                "<EXECUTE>commande</EXECUTE> (ex: <EXECUTE>spotify</EXECUTE>)."
            )

            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=MODEL_VISION,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt + "\n" + vision_instructions},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{jpg_as_text}"},
                            },
                        ],
                    }
                ],
                temperature=0.5, max_tokens=500,
            )
            
            text = completion.choices[0].message.content
            
            if "<EXECUTE>" in text:
                parts = text.split("<EXECUTE>")
                spoken_part = parts[0]
                if len(parts) > 1:
                    cmd = parts[1].split("</EXECUTE>")[0]
                    self.execute_system_command(cmd)
                    text = spoken_part + f" (Vision Action: {cmd})"
                else:
                    text = spoken_part

            self.text_out.emit(f"NEXUS (Vision): {text}")
            await self.speak(text)
            
        except Exception as e:
            self.logs_out.emit("ERR", f"Vision Error: {e}")

    def execute_system_command(self, cmd):
        self.logs_out.emit("CMD", f"Exécution : {cmd}")
        cmd = cmd.strip()
        if cmd.startswith("http") or "www." in cmd:
            try: webbrowser.open(cmd); return
            except: pass
        try:
            if sys.platform.startswith('win'): os.startfile(cmd)
            else: subprocess.run(['xdg-open', cmd])
        except Exception:
            try: subprocess.run(f'{cmd}', shell=True)
            except: self.logs_out.emit("ERR", f"Échec ouverture: {cmd}")

    # --- CAPTURE VIDÉO ---
    async def camera_loop(self):
        cap = None
        while self.running:
            if self.mode != MODE_CAMERA:
                if cap: cap.release(); cap = None
                await asyncio.sleep(0.5); continue

            if not cap: 
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame
                self.send_to_gui(frame)
            await asyncio.sleep(0.03) 

    async def screen_loop(self):
        while self.running:
            if self.mode != MODE_SCREEN: await asyncio.sleep(0.5); continue
            img = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self.current_frame = frame
            self.send_to_gui(frame)
            await asyncio.sleep(0.1) 

    def send_to_gui(self, frame):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, w * ch, QImage.Format_BGR888)
        self.image_out.emit(qimg.copy())

# ======================================================
# 3. GUI (Design Architecture de main.py)
# ======================================================
class ModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXUS AI - Groq Edition")
        self.resize(1300, 800)
        
        # --- STYLESHEET DU MAIN.PY (Copie exacte) ---
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
            QPushButton#mic_btn { background-color: #333; border-radius: 20px; }
            QPushButton#mic_btn:checked { background-color: #d32f2f; color: white; }
            QScrollBar:vertical { background: #1e1e1e; width: 8px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); main_layout.setContentsMargins(20, 20, 20, 20); main_layout.setSpacing(20)

        # --- COLONNE GAUCHE (CHAT + VISUALIZER + INPUT) ---
        left_col = QVBoxLayout()
        
        # Chat
        self.chat_display = QTextEdit(); self.chat_display.setObjectName("chat")
        self.chat_display.setReadOnly(True)
        chat_frame = QFrame(); chat_frame.setObjectName("panel"); chat_layout = QVBoxLayout(chat_frame)
        chat_layout.addWidget(self.chat_display)
        left_col.addWidget(chat_frame, 1)

        # Visualizer (Orange)
        self.visualizer = AudioVisualizer()
        left_col.addWidget(self.visualizer)

        # Input Area (Barre de saisie + Bouton Mic ajouté pour main2.py)
        inp_layout = QHBoxLayout()
        self.input_box = QLineEdit(); self.input_box.setPlaceholderText("Message Nexus AI...")
        self.input_box.returnPressed.connect(self.send_message)
        
        # Bouton Microphone
        self.btn_mic = QPushButton("🎤")
        self.btn_mic.setObjectName("mic_btn")
        self.btn_mic.setCheckable(True)
        self.btn_mic.setFixedSize(40, 40)
        self.btn_mic.clicked.connect(self.toggle_mic)
        
        inp_layout.addWidget(self.input_box)
        inp_layout.addWidget(self.btn_mic) 
        left_col.addLayout(inp_layout)
        
        main_layout.addLayout(left_col, 6)

        # --- COLONNE DROITE (VIDEO + BOUTONS + LOGS) ---
        right_col = QVBoxLayout()
        
        # Vidéo
        self.video_label = QLabel("Camera Off"); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 12px; color: #555;")
        self.video_label.setMinimumHeight(250)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.video_label)

        # Boutons Modes
        ctrl_layout = QHBoxLayout()
        self.btn_cam = QPushButton("CAMERA"); self.btn_cam.setCheckable(True)
        self.btn_scr = QPushButton("SCREEN"); self.btn_scr.setCheckable(True)
        self.btn_off = QPushButton("OFF"); self.btn_off.setCheckable(True); self.btn_off.setChecked(True)
        
        self.btn_cam.clicked.connect(lambda: self.switch_mode(MODE_CAMERA))
        self.btn_scr.clicked.connect(lambda: self.switch_mode(MODE_SCREEN))
        self.btn_off.clicked.connect(lambda: self.switch_mode(MODE_TEXT))
        
        ctrl_layout.addWidget(self.btn_cam); ctrl_layout.addWidget(self.btn_scr); ctrl_layout.addWidget(self.btn_off)
        right_col.addLayout(ctrl_layout)

        # Logs
        log_frame = QFrame(); log_frame.setObjectName("panel")
        log_layout = QVBoxLayout(log_frame)
        log_layout.addWidget(QLabel("SYSTEM LOGS", objectName="log_title"))
        self.logs_display = QTextEdit(); self.logs_display.setObjectName("logs"); self.logs_display.setReadOnly(True)
        log_layout.addWidget(self.logs_display)
        right_col.addWidget(log_frame, 1)

        main_layout.addLayout(right_col, 4)

        # --- CONNEXION BACKEND ---
        self.backend = Backend()
        self.backend.text_out.connect(self.append_text)
        self.backend.image_out.connect(self.update_video)
        self.backend.logs_out.connect(self.append_log)
        self.backend.voice_state.connect(self.visualizer.set_active)
        self.backend.stt_status.connect(self.update_mic_status)
        self.backend.start()

    # --- LOGIQUE INTERFACE ---
    def send_message(self):
        text = self.input_box.text().strip()
        if text:
            self.chat_display.append(f"<div style='color:#aaaaaa; margin-top:10px;'>YOU: {escape(text)}</div>")
            if self.backend.mode == MODE_TEXT:
                self.backend.loop.call_soon_threadsafe(self.backend.q_text.put_nowait, text)
            else:
                self.backend.loop.call_soon_threadsafe(self.backend.q_vision.put_nowait, text)
            self.input_box.clear()

    def toggle_mic(self):
        self.backend.stt_active = not self.backend.stt_active
        if not self.backend.stt_active:
             self.btn_mic.setChecked(False)

    @Slot(str)
    def update_mic_status(self, status):
        is_listening = self.backend.stt_active
        self.btn_mic.setChecked(is_listening)
        self.input_box.setPlaceholderText(f"Micro: {status}" if is_listening else "Message Nexus AI...")

    @Slot(str)
    def append_text(self, text):
        if text.startswith("NEXUS"):
            content = text.replace("NEXUS:", "").replace("NEXUS (Vision):", "")
            self.chat_display.append(f"<span style='color:#ffffff;'><b>NEXUS:</b> {escape(content)}</span>")

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
        self.btn_cam.setChecked(mode == MODE_CAMERA)
        self.btn_scr.setChecked(mode == MODE_SCREEN)
        self.btn_off.setChecked(mode == MODE_TEXT)
        self.backend.mode = mode
        self.append_log("MODE", f"Changement vers {mode}")

    def closeEvent(self, event):
        self.backend.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())
