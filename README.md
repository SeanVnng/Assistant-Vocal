# 🧠 NEXUS AI - Groq Edition

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Groq](https://img.shields.io/badge/Moteur-Groq%20LPU-orange?style=for-the-badge&logo=fastapi)
![Llama](https://img.shields.io/badge/Model-Llama%203.2%20Vision-blueviolet?style=for-the-badge)
![Qt](https://img.shields.io/badge/GUI-PySide6-green?style=for-the-badge&logo=qt)

**NEXUS AI** est un assistant de bureau multimodal ultra-rapide. Il utilise la puissance des **LPU Groq** pour une latence quasi nulle et intègre des modèles de vision **Llama 3.2** pour "voir" votre monde.

Contrairement aux assistants classiques, NEXUS peut analyser votre webcam ou votre écran en temps réel, exécuter des commandes système (ouvrir des applications, sites web) et converser vocalement, le tout via une interface graphique moderne et réactive.

---

## ✨ Fonctionnalités Principales

* **👁️ Vision Multimodale :**
    * **Mode Caméra :** Analyse de votre environnement physique via webcam (HD 720p).
    * **Mode Écran :** Capture et analyse de votre bureau pour vous aider sur vos tâches.
    * *Technologie :* Utilisation de `Llama-3.2-Vision` via l'API Groq pour une description instantanée.
* **⚡ Vitesse & Intelligence :**
    * Propulsé par **Groq** (Inférence IA la plus rapide du monde).
    * Réponses concises et pertinentes en Français.
* **🛠️ Contrôle Système :**
    * **Commandes Vocales/Texte :** Demandez *"Ouvre Spotify"* ou *"Lance Google"* et NEXUS s'exécute.
    * **Support :** Sites web (URL) et Applications locales (exe/binaires).
* **🗣️ Interaction Vocale :**
    * **STT (Écoute) :** Reconnaissance vocale Google (SpeechRecognition).
    * **TTS (Parole) :** Synthèse vocale locale rapide et sans latence (pyttsx3).
* **🖥️ Interface Moderne :**
    * GUI sombre (Dark Theme) avec accents orange.
    * Visualiseur audio dynamique.
    * Double affichage : Chat + Retour Vidéo/Logs.

---

## ⚙️ Prérequis

* **Python 3.10** ou supérieur.
* Une **Clé API Groq** (Gratuite et disponible sur [console.groq.com](https://console.groq.com)).
* Un microphone et une webcam.

---

## 🚀 Installation

1.  **Cloner le projet :**
    ```bash
    git clone [https://github.com/votre-username/nexus-ai.git](https://github.com/votre-username/nexus-ai.git)
    cd nexus-ai
    ```

2.  **Créer un environnement virtuel :**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

    *Contenu du `requirements.txt` :*
    ```text
    PySide6
    groq
    python-dotenv
    SpeechRecognition
    pyttsx3
    opencv-python
    numpy
    Pillow
    pyaudio
    ```

    > **Note :** Si vous rencontrez une erreur avec `pyaudio`, installez `pipwin install pyaudio` (Windows) ou `sudo apt install portaudio19-dev` (Linux).

4.  **Configuration (.env) :**
    Créez un fichier nommé `.env` à la racine et ajoutez votre clé :

    ```ini
    # Clé API Groq (Obligatoire)
    GROQ_API_KEY=gsk_votre_cle_ici...

    # Configuration Modèles (Optionnel, valeurs par défaut)
    MODEL_TEXT=meta-llama/llama-4-scout-17b-16e-instruct
    MODEL_VISION=meta-llama/llama-4-scout-17b-16e-instruct
    ```

---

## 🎮 Utilisation

Lancez simplement le script principal :

```bash
python final_nexus.py

Réalisé par Seann
