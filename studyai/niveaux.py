# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/niveaux.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : niveau 0 (LLM seul), la référence à battre
# ------------------------------------------------------------

PROMPT_NIVEAU_0 = """Tu es StudyAI, un assistant qui aide des étudiants d'école d'ingénieur à réviser leurs cours d'IA générative.
Règles :
- Réponds en français, de façon claire et courte (5 phrases maximum).
- Si tu n'es pas sûr de la réponse, dis-le honnêtement au lieu d'inventer.
- Si la demande est trop vague, pose une question pour la faire préciser.
- Si la demande n'a rien à voir avec les révisions, refuse poliment en expliquant que tu sers à réviser les cours."""


def niveau0(question, llm):
    """Niveau 0 : on pose la question au LLM avec un prompt soigné, sans documents ni outils."""
    messages = [
        {"role": "system", "content": PROMPT_NIVEAU_0},
        {"role": "user", "content": question},
    ]
    rep = llm.chat(messages)
    return {
        "niveau": 0,
        "question": question,
        "reponse": rep["contenu"],
        "sources": [],
        "passages": [],
        "tokens": rep["tokens"],
        "duree": rep["duree"],
        "etapes": [{"type": "llm", "tokens": rep["tokens"], "duree": rep["duree"]}],
    }
