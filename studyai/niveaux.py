# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/niveaux.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.1
# Rôle    : niveau 0 (LLM seul) et niveau 1 (RAG avec sources et abstention)
# ------------------------------------------------------------

from studyai import config
from studyai.recherche import formater_source

PHRASE_ABSTENTION = "Je ne trouve pas cette information dans tes cours."

PROMPT_NIVEAU_0 = """Tu es StudyAI, un assistant qui aide des étudiants d'école d'ingénieur à réviser leurs cours d'IA générative.
Règles :
- Réponds en français, de façon claire et courte (5 phrases maximum).
- Si tu n'es pas sûr de la réponse, dis-le honnêtement au lieu d'inventer.
- Si la demande est trop vague, pose une question pour la faire préciser.
- Si la demande n'a rien à voir avec les révisions, refuse poliment en expliquant que tu sers à réviser les cours."""

PROMPT_NIVEAU_1 = f"""Tu es StudyAI, un assistant qui aide des étudiants à réviser à partir de LEURS cours.
Tu reçois des extraits de cours, chacun précédé de sa source entre crochets.
Règles :
- Réponds UNIQUEMENT avec les informations des extraits, jamais avec tes propres connaissances.
- Après chaque information, cite la source entre crochets exactement comme dans les extraits, par exemple [cours_02_autoencodeurs_vae.pdf p.1].
- Si les extraits ne contiennent pas la réponse, réponds exactement : "{PHRASE_ABSTENTION}"
- Si la demande est trop vague, pose une question pour la faire préciser.
- Si la demande n'a rien à voir avec les révisions, refuse poliment.
- Réponds en français, de façon claire et courte (5 phrases maximum)."""


def formater_extraits(resultats):
    """Transforme les résultats de recherche en texte à donner au LLM, avec la source avant chaque extrait."""
    blocs = []
    for morceau, score in resultats:
        blocs.append(f"[{formater_source(morceau)}]\n{morceau['texte']}")
    return "\n\n".join(blocs)


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


def niveau1(question, llm, recherche, seuil=config.SEUIL_ABSTENTION, k=config.K_RECHERCHE):
    """Niveau 1 (RAG) : on cherche les passages des cours, puis le LLM répond en citant ses sources.

    Si aucun passage n'est assez proche de la question, on s'abstient sans même appeler le LLM.
    """
    resultats = recherche.chercher(question, k)
    etapes = [{
        "type": "recherche",
        "moteur": recherche.nom,
        "passages": [{"id": m["id"], "score": round(s, 3)} for m, s in resultats],
    }]
    passages = [m["id"] for m, s in resultats]

    meilleur_score = resultats[0][1] if resultats else 0.0
    if meilleur_score < seuil:
        etapes.append({"type": "abstention", "raison": f"meilleur score {meilleur_score:.2f} < seuil {seuil}"})
        return {
            "niveau": 1, "question": question, "reponse": PHRASE_ABSTENTION,
            "sources": [], "passages": passages, "tokens": 0, "duree": 0.0, "etapes": etapes,
        }

    messages = [
        {"role": "system", "content": PROMPT_NIVEAU_1},
        {"role": "user", "content": f"Extraits de cours :\n\n{formater_extraits(resultats)}\n\nQuestion : {question}"},
    ]
    rep = llm.chat(messages)
    etapes.append({"type": "llm", "tokens": rep["tokens"], "duree": rep["duree"]})
    return {
        "niveau": 1,
        "question": question,
        "reponse": rep["contenu"],
        "sources": sorted({formater_source(m) for m, s in resultats}),
        "passages": passages,
        "tokens": rep["tokens"],
        "duree": rep["duree"],
        "etapes": etapes,
    }
