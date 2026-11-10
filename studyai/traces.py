# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/traces.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : enregistre la trace de chaque exécution en JSONL
#           (étapes, outils, arguments, tokens, durée) pour l'analyse d'erreurs
# ------------------------------------------------------------

import json
from datetime import datetime
from pathlib import Path

from studyai import config


def ecrire_trace(resultat, fichier=config.FICHIER_TRACES):
    """Ajoute une exécution à la fin du fichier de traces (une ligne JSON = une exécution)."""
    ligne = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "niveau": resultat.get("niveau"),
        "question": resultat.get("question"),
        "reponse": resultat.get("reponse"),
        "statut": resultat.get("statut", "termine"),
        "sources": resultat.get("sources", []),
        "etapes": resultat.get("etapes", []),
        "tokens": resultat.get("tokens", 0),
        "duree": round(resultat.get("duree", 0.0), 3),
    }
    if "erreur" in resultat:
        ligne["erreur"] = resultat["erreur"]
    Path(fichier).parent.mkdir(parents=True, exist_ok=True)
    with open(fichier, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


def lire_traces(fichier=config.FICHIER_TRACES, nombre=20):
    """Renvoie les dernières traces (les plus récentes en premier) pour les afficher dans l'interface."""
    chemin = Path(fichier)
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        lignes = [json.loads(l) for l in f if l.strip()]
    return lignes[-nombre:][::-1]
