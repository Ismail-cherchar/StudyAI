# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/config.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : regroupe tous les réglages du projet (lus dans le .env)
# ------------------------------------------------------------

import os
from pathlib import Path

from dotenv import load_dotenv

# dossier racine du projet (celui qui contient app.py et eval.py)
RACINE = Path(__file__).resolve().parent.parent

load_dotenv(RACINE / ".env")

# --- LLM : n'importe quelle API compatible OpenAI (Ollama, Groq, Mistral...) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Embeddings : petit modèle multilingue qui tourne sur CPU ---
MODELE_EMBEDDINGS = os.getenv(
    "MODELE_EMBEDDINGS", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Dossiers ---
DOSSIER_COURS = RACINE / "data" / "cours"
DOSSIER_INDEX = RACINE / "data" / "index"
FICHIER_TESTS = RACINE / "data" / "tests.jsonl"
DOSSIER_SAUVEGARDES = RACINE / "sauvegardes"
DOSSIER_RESULTATS = RACINE / "resultats"
FICHIER_TRACES = RACINE / "traces" / "traces.jsonl"

# --- Découpage des cours ---
# le modèle d'embeddings lit au maximum 128 tokens, donc on garde des morceaux
# d'environ 80 mots (~110 tokens en français) pour ne pas couper la fin du texte
TAILLE_MORCEAU = 80
CHEVAUCHEMENT = 20

# --- Recherche ---
K_RECHERCHE = 5
# en dessous de cette similarité cosinus, on considère que les cours ne parlent pas du sujet
SEUIL_ABSTENTION = float(os.getenv("SEUIL_ABSTENTION", "0.35"))

# --- Agent : la boucle est bornée ---
MAX_ETAPES_AGENT = 6
MAX_TOKENS_AGENT = 12000
