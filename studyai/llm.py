# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/llm.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : envoie les messages au LLM avec une simple requête HTTP
#           (pas de framework, juste l'API /chat/completions)
# ------------------------------------------------------------

import time

import requests

from studyai import config


class ErreurLLM(Exception):
    """Erreur levée quand le LLM ne répond pas (serveur éteint, mauvaise clé...)."""


def estimer_tokens(texte):
    """Estime le nombre de tokens d'un texte (environ 4 caractères par token)."""
    return max(1, len(texte or "") // 4)


class LLM:
    """Petit client pour une API de chat compatible OpenAI (Ollama, Groq, Mistral...)."""

    def __init__(self, base_url=None, modele=None, cle_api=None, temperature=None):
        """Garde l'adresse de l'API, le nom du modèle, la clé et la température."""
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.modele = modele or config.LLM_MODEL
        self.cle_api = cle_api or config.LLM_API_KEY
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    def chat(self, messages, outils=None, max_tokens=800):
        """Envoie la conversation au LLM et renvoie le texte, les appels d'outils, les tokens et la durée."""
        corps = {
            "model": self.modele,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if outils:
            corps["tools"] = outils

        debut = time.time()
        try:
            reponse = requests.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.cle_api},
                json=corps,
                timeout=180,
            )
        except requests.exceptions.ConnectionError:
            raise ErreurLLM(f"Impossible de joindre le LLM à l'adresse {self.base_url} (Ollama est-il lancé ?).")
        except requests.exceptions.RequestException as e:
            raise ErreurLLM(f"Problème pendant l'appel au LLM : {e}")
        if reponse.status_code != 200:
            raise ErreurLLM(f"Le LLM a renvoyé l'erreur {reponse.status_code} : {reponse.text[:300]}")

        data = reponse.json()
        duree = time.time() - debut
        message = data["choices"][0]["message"]
        contenu = message.get("content") or ""

        # certaines API ne renvoient pas "usage", dans ce cas on fait une estimation
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        if not tokens:
            tokens = sum(estimer_tokens(str(m.get("content"))) for m in messages) + estimer_tokens(contenu)

        return {
            "contenu": contenu,
            "appels": message.get("tool_calls") or [],
            "tokens": tokens,
            "duree": round(duree, 3),
        }
