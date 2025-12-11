import sys
import os 
import io
import base64
import asyncio
import threading
import subprocess 
import webbrowser
from html import escape
import json 
import time

# --- IMPORTS AUDIO ET ASYNCHRONES NÉCESSAIRES ---
import websockets 
import pyaudio 
from dotenv import load_dotenv 

# --- GUI Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLabel,
    QVBoxLayout, QWidget, QLineEdit, QHBoxLayout,
    QPushButton, QFrame
)
from PySide6.QtCore import Qt, Slot, Signal, QObject, QTimer, QThread 
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QPainter, QColor, QBrush 

import cv2
import numpy as np
from PIL import ImageGrab, Image as PILImage

from google import genai
from google.genai.types import Image, Part 


# --- CONFIG & KEYS (Chargement depuis .env) ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID", 'pFZP5JQG7iQjIQuC4Bku') 

if not GEMINI_API_KEY or not ELEVENLABS_API_KEY:
    sys.exit("ERREUR: Les clés API (GEMINI_API_KEY ou ELEVENLABS_API_KEY) sont manquantes dans le fichier .env ou l'environnement.")


MODEL_ID = "gemini-2.5-flash"

# Configuration Audio
RECV_RATE = 24000 # Sample rate pour la sortie TTS
MIC_RATE = 16000  # Sample rate pour l'entrée Microphone (Standard pour STT)
CHUNK_SIZE = 1024 # Taille des fragments pour PyAudio

MODE_TEXT = 0
MODE_CAMERA = 1
MODE_SCREEN = 2

# ==============================================================================
# 1. ANIMATION AUDIO
# ==============================================================================
class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.bars = 10
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
            self.values = [max(0.1, min(1.0, v + (np.random.random() - 0.5) * 0.5)) for v in self.values]
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
# ===================== BACKEND (RTSTT) ================
# ======================================================

class Backend(QObject):
    text_out = Signal(str)
    image_out = Signal(QImage)
    logs_out = Signal(str, str)
    voice_state = Signal(bool) 
    stt_out = Signal(str) # Texte final reconnu
    stt_status = Signal(str) # État du STT (Connexion/Écoute/Déconnexion)
    stt_partial = Signal(str) # Texte partiel en temps réel

    def __init__(self):
        super().__init__()
        self.mode = MODE_TEXT
        self.running = True
        self.current_frame = None  
        self.stt_active = False # Contrôle le streaming du microphone

        self.loop = asyncio.new_event_loop()
        self.client = genai.Client(api_key=GEMINI_API_KEY) 
        self.pya = pyaudio.PyAudio() 

        self.q_text = asyncio.Queue()
        self.q_vision = asyncio.Queue()  
        self.q_tts_in = asyncio.Queue()    
        self.q_audio_out = asyncio.Queue() 
        self.q_mic_out = asyncio.Queue() # Nouvelle file pour les données microphone
        
        self.last_send_time = 0
        self.stt_websocket_task = None

    def start(self):
        threading.Thread(
            target=lambda: self.loop.run_until_complete(self.main_loop()),
            daemon=True
        ).start()

    def stop(self):
        self.running = False
        
    # ======================================================
    # ========== TASK MIC INPUT (PyAudio) ==================
    # ======================================================
    async def task_mic_input(self):
        """Capture les données du microphone et les envoie à la file d'attente."""
        try:
            # Démarrer le stream PyAudio pour l'entrée
            stream = self.pya.open(
                format=pyaudio.paInt16, 
                channels=1, 
                rate=MIC_RATE, 
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            self.logs_out.emit("INFO", "Microphone stream démarré.")
            
            while self.running:
                # Lire les données du microphone de manière non bloquante
                if self.stt_active:
                    data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                    await self.q_mic_out.put(data)
                else:
                    await asyncio.sleep(0.1) # Attend si le STT est désactivé
                    
        except Exception as e:
            self.logs_out.emit("CRITICAL", f"Erreur PyAudio Input: {e}")
        finally:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()

    # ======================================================
    # ========== TASK STT WEBSOCKET (REALTIME) =============
    # ======================================================
    async def task_stt_websocket(self):
        """Envoie l'audio du micro à un service STT en streaming et reçoit le texte."""
        # NOTE: Ceci est un PROTOCOLE HYPOTHÉTIQUE, car nous n'avons pas d'API RTSTT standard.
        # Le protocole ElevenLabs STT n'est pas public. Nous utilisons ici une structure générique.
        
        # URI ElevenLabs (hypothétique RTSTT)
        # Note: L'API publique d'ElevenLabs pour le STT n'est pas par WebSocket pour l'instant.
        # Si vous utilisez Google Cloud STT, l'URI et le protocole seraient différents.
        uri = "wss://rtstt.service-example.com/ws" 
        
        self.stt_status.emit("Prêt pour STT. Appuyez pour parler.")

        while self.running:
            if not self.stt_active:
                await asyncio.sleep(0.5)
                continue
                
            self.stt_status.emit("Connexion STT...")
            
            try:
                # Début de la session WebSocket
                async with websockets.connect(uri) as ws:
                    self.stt_status.emit("Écoute en cours...")
                    self.logs_out.emit("INFO", "STT: WebSocket Connecté. Début du streaming.")
                    
                    # Tâche pour envoyer l'audio
                    async def send_mic_data():
                        # Envoyer les paramètres audio initiaux
                        await ws.send(json.dumps({
                            "type": "config", 
                            "api_key": ELEVENLABS_API_KEY,
                            "rate": MIC_RATE, 
                            "language": "fr-FR"
                        }))
                        
                        while self.stt_active and self.running:
                            try:
                                # Envoie les fragments audio PyAudio
                                data = await self.q_mic_out.get()
                                
                                # Envoi des données audio encodées en base64
                                await ws.send(json.dumps({
                                    "type": "audio", 
                                    "data": base64.b64encode(data).decode('utf-8')
                                }))
                                self.q_mic_out.task_done()
                            except asyncio.QueueEmpty:
                                await asyncio.sleep(0.01)
                            except websockets.exceptions.ConnectionClosedOK:
                                break
                            
                        # Signal de fin de transmission
                        await ws.send(json.dumps({"type": "end"}))
                        
                    # Tâche pour recevoir le texte
                    async def receive_text():
                        nonlocal last_text
                        last_text = ""
                        while self.stt_active and self.running:
                            try:
                                message = await ws.recv()
                                data = json.loads(message)
                                
                                if data.get("text_partial"):
                                    self.stt_partial.emit(data["text_partial"])
                                    
                                if data.get("text_final"):
                                    final_text = data["text_final"]
                                    self.stt_out.emit(final_text)
                                    self.logs_out.emit("INFO", f"STT Final: {final_text}")
                                    
                                    # Déclenche la requête Gemini et arrête l'enregistrement
                                    self.loop.call_soon_threadsafe(self.q_text.put_nowait, final_text)
                                    self.stt_active = False # Arrêt du streaming
                                    break
                                    
                            except websockets.exceptions.ConnectionClosedOK:
                                break
                            except Exception as e:
                                self.logs_out.emit("ERR", f"STT Receive Error: {e}")
                                break
                    
                    # Exécuter les deux tâches simultanément
                    await asyncio.gather(send_mic_data(), receive_text())
                    
            except websockets.exceptions.ConnectionRefused as e:
                self.logs_out.emit("ERR", f"STT Connexion refusée. Le serveur est-il actif ? {e}")
            except Exception as e: 
                self.logs_out.emit("ERR", f"STT WebSocket Fatal Error: {e}")
            finally: 
                self.stt_active = False
                self.stt_status.emit("Prêt pour STT. Appuyez pour parler.")

    # ======================================================
    # ========== TASK TTS (Websocket - Streaming) ==========
    # ... (inchangé)
    # ======================================================
    async def task_tts(self):
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_24000"
        while self.running:
            text = await self.q_tts_in.get()
            if not text: 
                self.q_tts_in.task_done()
                continue
            
            self.logs_out.emit("INFO", "TTS: Connexion WebSocket...")
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
                        
            except Exception as e: 
                self.logs_out.emit("ERR", f"TTS WebSocket Error: {e}")
            finally: 
                self.voice_state.emit(False)
                self.q_tts_in.task_done()

    # ======================================================
    # ========== TASK AUDIO OUTPUT (PyAudio) ===============
    # ======================================================
    async def task_audio_output(self):
        try:
            stream = self.pya.open(format=pyaudio.paInt16, channels=1, rate=RECV_RATE, output=True)
            while self.running:
                data = await self.q_audio_out.get()
                await asyncio.to_thread(stream.write, data)
                self.q_audio_out.task_done()
        except Exception as e:
            self.logs_out.emit("CRITICAL", f"PyAudio Output Error: {e}. PyAudio installé ?")

    # ======================================================
    # ========== MAIN BACKEND LOOP (MAJ AUDIO) =============
    # ======================================================
    async def main_loop(self):
        self.logs_out.emit("INFO", "Backend ready")

        # Démarrage des tâches asynchrones (Camera/Screen, TTS, Audio Output)
        asyncio.create_task(self.camera_loop())
        asyncio.create_task(self.screen_loop())
        asyncio.create_task(self.task_tts())
        asyncio.create_task(self.task_audio_output())
        asyncio.create_task(self.task_mic_input()) # Démarrage de la capture micro
        # Lancement du STT WebSocket
        self.stt_websocket_task = asyncio.create_task(self.task_stt_websocket()) 

        SYSTEM_PROMPT_COMMAND = (
            "Vous êtes un assistant capable d'exécuter des commandes système. "
            "Si l'utilisateur demande d'ouvrir une application (comme 'ouvrir Spotify') ou "
            "un site web (comme 'ouvrir google.com'), vous DEVEZ répondre UNIQUEMENT avec "
            "le format: <EXECUTE>cible</EXECUTE> (où 'cible' est l'application ou l'URL à ouvrir, "
            "ex: <EXECUTE>spotify</EXECUTE> ou <EXECUTE>https://www.google.com</EXECUTE>). "
            "Pour toute autre question, répondez normalement."
        )

        while self.running:
            # GESTION DES REQUÊTES TEXTE
            if not self.q_text.empty():
                user_text = await self.q_text.get()
                
                contents = [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT_COMMAND}]},
                    {"role": "user", "parts": [{"text": user_text}]}
                ]
                
                result_text = await asyncio.to_thread(
                    self.model_call,
                    contents
                )

                # Traitement de la commande
                if result_text.startswith("<EXECUTE>") and result_text.endswith("</EXECUTE>"):
                    command_to_exec = result_text.strip()[len("<EXECUTE>"): -len("</EXECUTE>")].strip()
                    execution_status = self.execute_system_command(command_to_exec)
                    
                    final_output = f"Exécution demandée... Statut: {execution_status}"
                    self.text_out.emit(f"<span style='color: orange;'>[COMMANDE SYSTÈME] {final_output}</span>")
                    
                else:
                    # Réponse normale
                    self.text_out.emit(result_text)
                    self.loop.call_soon_threadsafe(self.q_tts_in.put_nowait, result_text) 
                    
                self.q_text.task_done()
            
            # GESTION DES REQUÊTES VISION
            if not self.q_vision.empty():
                user_prompt = await self.q_vision.get()
                await self.analyze_image(user_prompt)
                self.q_vision.task_done()

            await asyncio.sleep(0.05)


    # ======================================================
    # ========== AUTRES MÉTHODES DU BACKEND (INCHANGÉES) ===
    # ======================================================
    # ... (camera_loop, screen_loop, execute_system_command, model_call, analyze_image) ...
    async def camera_loop(self):
        cap = None
        while self.running:
            if self.mode != MODE_CAMERA:
                await asyncio.sleep(0.2)
                continue

            if cap is None or not cap.isOpened():
                for i in range(3):
                    cam = cv2.VideoCapture(i)
                    if cam.isOpened():
                        ret, test = cam.read()
                        if ret:
                            cap = cam
                            break
                        cam.release()

                if cap is None:
                    await asyncio.sleep(1)
                    continue

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                cap.release()
                cap = None
                await asyncio.sleep(0.2)
                continue

            self.send_to_gui(frame)
            self.current_frame = frame
            
            await asyncio.sleep(0.05)

    async def screen_loop(self):
        while self.running:
            if self.mode != MODE_SCREEN:
                await asyncio.sleep(0.2)
                continue

            img = await asyncio.to_thread(ImageGrab.grab)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            self.send_to_gui(frame)
            self.current_frame = frame
            
            await asyncio.sleep(0.1)

    def execute_system_command(self, command: str):
        self.logs_out.emit("INFO", f"Tentative d'exécution de: {command}")
        
        try:
            if sys.platform.startswith('win'):
                subprocess.run(f'start {command}', shell=True, check=True)
            elif sys.platform == 'darwin': 
                subprocess.run(['open', command], check=True)
            else: 
                subprocess.run(['xdg-open', command], check=True)
            
            return f"Commande '{command}' exécutée avec succès (tentative de lancement)."
                
        except subprocess.CalledProcessError as e:
            return f"[ERREUR SYSTÈME] Échec de la commande: {e}"
        except FileNotFoundError:
             return f"[ERREUR SYSTÈME] Application ou commande non trouvée: {command}"
        except Exception as e:
            return f"[ERREUR SYSTÈME] Erreur inattendue: {e}"

    def model_call(self, contents):
        try:
            resp = self.client.models.generate_content(
                model=MODEL_ID,
                contents=contents
            )
            return resp.text if hasattr(resp, "text") else str(resp)

        except Exception as e:
            return f"[API ERROR] {str(e)}"

    async def analyze_image(self, prompt: str): 
        if self.current_frame is None:
            self.logs_out.emit("WARN", "Pas d'image actuelle pour l'analyse.")
            return

        try:
            frame = self.current_frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            
            pil.thumbnail((800, 800)) 
            
            buf = io.BytesIO()
            pil.save(buf, format="JPEG") 
            img_bytes = buf.getvalue()

            img_part = Part.from_bytes(
                data=img_bytes,
                mime_type='image/jpeg' 
            )

            result = await asyncio.to_thread(
                self.model_call,
                [img_part, prompt] 
            )

            self.text_out.emit(result)
            self.loop.call_soon_threadsafe(self.q_tts_in.put_nowait, result) 

        except Exception as e:
            self.logs_out.emit("ERR", str(e))

    def send_to_gui(self, frame):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, w * ch, QImage.Format_BGR888)
        self.image_out.emit(qimg.copy())
            
# ======================================================
# ======================= GUI ==========================
# ======================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXUS – Realtime STT Intégration")
        self.resize(1300, 800)

        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)

        # LEFT (Chat)
        left = QVBoxLayout()

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        left.addWidget(self.chat)
        
        # Ajout du Visualizer audio
        self.visualizer = AudioVisualizer()
        left.addWidget(self.visualizer)

        # Input/STT Box
        input_stt_box = QVBoxLayout()

        # Champ de texte (inchangé)
        text_input_box = QHBoxLayout()
        self.input = QLineEdit()
        self.input.returnPressed.connect(self.send_message)
        btn_send = QPushButton("Envoyer")
        btn_send.clicked.connect(self.send_message)
        text_input_box.addWidget(self.input)
        text_input_box.addWidget(btn_send)
        
        input_stt_box.addLayout(text_input_box)
        
        # Bouton STT (Nouveau)
        self.btn_stt = QPushButton("🎙️ Démarrer Écoute (RTSTT)")
        self.btn_stt.clicked.connect(self.toggle_stt)
        self.btn_stt.setEnabled(True)
        input_stt_box.addWidget(self.btn_stt)
        
        # Champ pour le texte partiel
        self.lbl_stt_partial = QLabel("Texte Partiel: ...")
        self.lbl_stt_partial.setStyleSheet("color: gray; margin-top: -5px;")
        input_stt_box.addWidget(self.lbl_stt_partial)

        left.addLayout(input_stt_box)
        main.addLayout(left, 6)

        # RIGHT (Video + Logs + Modes)
        right = QVBoxLayout()

        self.video = QLabel("Aucune source")
        self.video.setStyleSheet("background:black; color:#bbb;")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumHeight(300)
        right.addWidget(self.video)

        mode_bar = QHBoxLayout()

        self.btn_text = QPushButton("TEXTE")
        self.btn_text.setCheckable(True)
        self.btn_text.setChecked(True)
        self.btn_text.clicked.connect(lambda: self.set_mode(MODE_TEXT))

        self.btn_cam = QPushButton("CAMERA")
        self.btn_cam.setCheckable(True)
        self.btn_cam.clicked.connect(lambda: self.set_mode(MODE_CAMERA))

        self.btn_screen = QPushButton("ÉCRAN")
        self.btn_screen.setCheckable(True)
        self.btn_screen.clicked.connect(lambda: self.set_mode(MODE_SCREEN))

        mode_bar.addWidget(self.btn_text)
        mode_bar.addWidget(self.btn_cam)
        mode_bar.addWidget(self.btn_screen)

        right.addLayout(mode_bar)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        right.addWidget(self.logs)

        main.addLayout(right, 4)

        # Backend
        self.backend = Backend()
        self.backend.text_out.connect(self.on_ai_output)
        self.backend.image_out.connect(self.update_video)
        self.backend.logs_out.connect(self.add_log)
        self.backend.voice_state.connect(self.visualizer.set_active)
        self.backend.stt_out.connect(self.on_stt_final_output) 
        self.backend.stt_partial.connect(self.on_stt_partial_output) # Affichage du texte en temps réel
        self.backend.stt_status.connect(self.update_stt_button) # Mise à jour du bouton
        self.backend.start()

    # --- NOUVELLES MÉTHODES RTSTT ---
    def toggle_stt(self):
        """Active/Désactive l'enregistrement du microphone pour le streaming STT."""
        self.backend.stt_active = not self.backend.stt_active
        self.update_stt_button(self.backend.stt_status.current_text() if hasattr(self.backend.stt_status, 'current_text') else "Prêt...")

    @Slot(str)
    def on_stt_partial_output(self, text):
        """Affiche le texte partiel reçu pendant le streaming."""
        self.lbl_stt_partial.setText(f"Texte Partiel: {text}...")

    @Slot(str)
    def on_stt_final_output(self, text):
        """Gère le texte final reconnu."""
        self.chat.append(f"<b>VOUS (Vocal Stream):</b> {escape(text)}")
        self.lbl_stt_partial.setText("Texte Partiel: ...")
        # Le backend s'occupe de mettre le texte dans la queue Gemini, pas besoin de le faire ici.

    @Slot(str)
    def update_stt_button(self, status):
        """Met à jour le texte et le style du bouton STT."""
        if self.backend.stt_active:
            self.btn_stt.setText(f"🔴 ÉCOUTE EN COURS (STOP)")
            self.btn_stt.setStyleSheet("background-color: red; color: white;")
        else:
            self.btn_stt.setText("🎙️ Démarrer Écoute (RTSTT)")
            self.btn_stt.setStyleSheet("")
            
        self.logs.append(f"[STT Status] {status}")


    # MODE SWITCH
    def set_mode(self, mode):
        self.backend.mode = mode

        self.btn_text.setChecked(mode == MODE_TEXT)
        self.btn_cam.setChecked(mode == MODE_CAMERA)
        self.btn_screen.setChecked(mode == MODE_SCREEN)

        # Désactive le RTSTT si on n'est pas en mode TEXTE
        if mode != MODE_TEXT and self.backend.stt_active:
            self.toggle_stt()

        if mode == MODE_TEXT:
            self.video.setText("Aucune source")
            self.btn_stt.setEnabled(True)
        else:
            self.video.setText("Caméra…" if mode == MODE_CAMERA else "Capture écran…")
            self.btn_stt.setEnabled(False)

    # MESSAGE SEND
    def send_message(self):
        msg = self.input.text().strip()
        if not msg:
            return
            
        self.chat.append(f"<b>YOU:</b> {escape(msg)}")

        if self.backend.mode == MODE_TEXT:
            self.backend.loop.call_soon_threadsafe(self.backend.q_text.put_nowait, msg)
        else:
            self.backend.loop.call_soon_threadsafe(self.backend.q_vision.put_nowait, msg)

        self.input.clear()

    @Slot(str)
    def on_ai_output(self, text):
        if text.startswith("<span style='color: orange;'>"):
             self.chat.append(text)
        else:
            self.chat.append(f"<span style='color:#4af;'>{escape(text)}</span>")

    @Slot(str, str)
    def add_log(self, t, c):
        self.logs.append(f"[{t}] {c}")

    @Slot(QImage)
    def update_video(self, img):
        pix = QPixmap.fromImage(img)
        self.video.setPixmap(
            pix.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
    
    def closeEvent(self, event):
        self.backend.stop()
        event.accept()


# APP
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
