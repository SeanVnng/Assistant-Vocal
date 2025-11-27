# 🧠 NEXUS AI - Advanced Multimodal Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Gemini](https://img.shields.io/badge/Model-Gemini%202.0%20Flash-orange?style=for-the-badge&logo=google)
![ElevenLabs](https://img.shields.io/badge/TTS-ElevenLabs-white?style=for-the-badge)
![Qt](https://img.shields.io/badge/GUI-PySide6-green?style=for-the-badge&logo=qt)

**NEXUS AI** est un assistant de bureau de nouvelle génération. Il ne se contente pas de vous écouter : **il voit ce que vous voyez**. Grâce à l'intégration profonde de l'API **Google Gemini Live** et d'**OpenCV**, NEXUS peut analyser votre flux webcam ou votre écran en temps réel tout en conversant naturellement avec une voix ultra-réaliste via **ElevenLabs**.

---

## ✨ Fonctionnalités Principales

* **👁️ Vision Temps Réel :**
    * **Mode Caméra :** L'IA voit votre environnement physique (Scan automatique des ports caméra).
    * **Mode Écran :** L'IA regarde votre écran pour vous aider à coder, lire des documents ou naviguer.
    * *Instruction Système Avancée :* Gemini est configuré pour analyser les flux vidéo en continu.
* **🗣️ Conversation Fluide :**
    * Latence ultra-faible grâce aux WebSockets.
    * Voix réaliste et expressive (ElevenLabs Turbo v2.5).
    * Visualiseur audio dynamique dans l'interface.
* **🛠️ Outils Système & Agents :**
    * **Fichiers :** Création, lecture et modification de fichiers/dossiers.
    * **Navigation :** Recherche Google et ouverture de sites web.
    * **Apps :** Lancement d'applications de bureau.
    * **Code :** Exécution et analyse de code Python.
* **🖥️ Interface Moderne :**
    * GUI sombre et minimaliste (Dark Theme).
    * Double affichage : Chat utilisateur vs Logs système (pour voir ce que l'IA fait en arrière-plan).

---

## ⚙️ Prérequis

Avant de commencer, assurez-vous d'avoir installé :

* **Python 3.10** ou supérieur.
* **Clé API Google Gemini** (Google AI Studio).
* **Clé API ElevenLabs** (Pour la synthèse vocale).
* *(Optionnel pour la version Wake-Word)* **Clé Picovoice Access Key**.

---

## 🚀 Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/votre-username/nexus-ai.git](https://github.com/votre-username/nexus-ai.git)
    cd nexus-ai
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    Créez un fichier `requirements.txt` avec le contenu ci-dessous, puis installez-le :
    
    *Contenu de `requirements.txt` :*
    ```text
    google-genai
    python-dotenv
    opencv-python
    pyaudio
    Pillow
    PySide6
    websockets
    numpy
    pvporcupine
    ```

    **Commande d'installation :**
    ```bash
    pip install -r requirements.txt
    ```

    > **Note pour Linux :** Vous devrez peut-être installer `portaudio19-dev` (`sudo apt install portaudio19-dev`) pour PyAudio.

4.  **Configuration des clés API :**
    Créez un fichier `.env` à la racine du projet et remplissez-le comme suit :

    ```ini
    GEMINI_API_KEY=votre_cle_gemini_ici
    ELEVENLABS_API_KEY=votre_cle_elevenlabs_ici
    
    # Requis uniquement pour la version test2.py (Wake Word)
    PICOVOICE_API_KEY=votre_cle_picovoice_ici
    ```

---

## 🎮 Utilisation

### Lancer la version principale (GUI Moderne)
C'est la version recommandée avec l'interface sombre, les logs système et la gestion avancée de la vision.

```bash
python main.py
