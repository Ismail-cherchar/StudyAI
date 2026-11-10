# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/outils.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : les outils de l'agent (décrits par un schéma JSON),
#           classés par effet (lecture / écriture), avec les règles
#           métier vérifiées par du code
# ------------------------------------------------------------

import json
import re
from datetime import datetime
from pathlib import Path

from studyai import config
from studyai.recherche import formater_source

# description des outils envoyée au LLM (format "tools" de l'API OpenAI)
SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher_cours",
            "description": "Cherche dans les cours PDF les passages utiles pour une requête. "
                           "À utiliser avant de répondre à une question de cours ou d'écrire une fiche.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {"type": "string", "description": "ce que l'on cherche, en quelques mots"}
                },
                "required": ["requete"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "construire_qcm",
            "description": "Construit un QCM corrigé (4 choix par question) à partir des cours sur un sujet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sujet": {"type": "string", "description": "le sujet du QCM, par exemple 'les GAN'"},
                    "nb_questions": {"type": "integer", "description": "entre 1 et 10"},
                },
                "required": ["sujet", "nb_questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lister_sauvegardes",
            "description": "Liste les fiches et les résultats de QCM déjà enregistrés.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enregistrer_fiche",
            "description": "ÉCRITURE : enregistre une fiche de révision. L'utilisateur devra confirmer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "titre": {"type": "string"},
                    "contenu": {"type": "string", "description": "notions clés, définitions et un exemple"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "sources utilisées, par exemple ['cours_03_gan.pdf p.1']",
                    },
                },
                "required": ["titre", "contenu", "sources"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enregistrer_resultat",
            "description": "ÉCRITURE : enregistre le score obtenu à un QCM. L'utilisateur devra confirmer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sujet": {"type": "string"},
                    "score": {"type": "integer", "description": "nombre de bonnes réponses"},
                    "total": {"type": "integer", "description": "nombre total de questions"},
                },
                "required": ["sujet", "score", "total"],
            },
        },
    },
]

# effet de chaque outil : les outils d'écriture attendent une confirmation
EFFETS = {
    "rechercher_cours": "lecture",
    "construire_qcm": "lecture",
    "lister_sauvegardes": "lecture",
    "enregistrer_fiche": "ecriture",
    "enregistrer_resultat": "ecriture",
}

PROMPT_QCM = """Tu crées des QCM de révision à partir d'extraits de cours.
Réponds UNIQUEMENT avec une liste JSON, sans texte autour, de la forme :
[{"question": "...", "choix": ["...", "...", "...", "..."], "reponse": 0, "explication": "..."}]
- exactement 4 choix par question, une seule bonne réponse ;
- "reponse" est l'indice (0 à 3) de la bonne réponse dans "choix" ;
- les questions et les explications viennent uniquement des extraits ;
- tout est en français."""


class ErreurOutil(Exception):
    """Erreur levée quand une règle métier n'est pas respectée (le message est renvoyé au LLM)."""


def lire_liste_json(texte):
    """Récupère la liste JSON dans la réponse du LLM (il ajoute parfois du texte ou des ``` autour)."""
    debut = texte.find("[")
    fin = texte.rfind("]")
    if debut == -1 or fin == -1:
        raise ErreurOutil("Le QCM généré n'est pas une liste JSON.")
    try:
        return json.loads(texte[debut:fin + 1])
    except json.JSONDecodeError:
        raise ErreurOutil("Le QCM généré n'est pas un JSON valide.")


def valider_questions(questions, nb_attendu):
    """Vérifie chaque question du QCM : un énoncé, 4 choix et un indice de bonne réponse correct."""
    if not isinstance(questions, list) or len(questions) < nb_attendu:
        raise ErreurOutil(f"Le QCM doit contenir {nb_attendu} questions.")
    propres = []
    for q in questions[:nb_attendu]:
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            raise ErreurOutil("Une question du QCM n'a pas d'énoncé.")
        choix = q.get("choix")
        if not isinstance(choix, list) or len(choix) != 4 or not all(str(c).strip() for c in choix):
            raise ErreurOutil("Chaque question doit avoir exactement 4 choix non vides.")
        try:
            reponse = int(q.get("reponse"))
        except (TypeError, ValueError):
            raise ErreurOutil("L'indice de la bonne réponse doit être un entier.")
        if not 0 <= reponse <= 3:
            raise ErreurOutil("L'indice de la bonne réponse doit être entre 0 et 3.")
        propres.append({
            "question": str(q["question"]).strip(),
            "choix": [str(c).strip() for c in choix],
            "reponse": reponse,
            "explication": str(q.get("explication", "")).strip(),
        })
    return propres


class Outils:
    """Regroupe les outils de l'agent et le dossier où on enregistre les fiches et les résultats."""

    def __init__(self, recherche, llm, dossier_sauvegardes=config.DOSSIER_SAUVEGARDES,
                 seuil=config.SEUIL_ABSTENTION):
        """Garde le moteur de recherche, le LLM (pour les QCM) et le dossier de sauvegarde."""
        self.recherche = recherche
        self.llm = llm
        self.dossier = Path(dossier_sauvegardes)
        self.seuil = seuil
        self.sources_connues = {m["source"] for m in recherche.morceaux}
        self.qcm_construits = []
        self.tokens_outils = 0  # tokens consommés par les outils eux-mêmes (le QCM appelle le LLM)

    def schemas(self):
        """Renvoie la description JSON des outils pour le LLM."""
        return SCHEMAS

    def est_sensible(self, nom):
        """Vrai si l'outil écrit quelque chose : il faudra demander une confirmation."""
        return EFFETS.get(nom) == "ecriture"

    def executer(self, nom, arguments):
        """Lance l'outil demandé. Les erreurs ne plantent pas l'agent : elles sont renvoyées au modèle."""
        fonctions = {
            "rechercher_cours": self.rechercher_cours,
            "construire_qcm": self.construire_qcm,
            "lister_sauvegardes": self.lister_sauvegardes,
            "enregistrer_fiche": self.enregistrer_fiche,
            "enregistrer_resultat": self.enregistrer_resultat,
        }
        if nom not in fonctions:
            return {"erreur": f"Outil inconnu : {nom}"}
        try:
            return fonctions[nom](**arguments)
        except ErreurOutil as e:
            return {"erreur": str(e)}
        except TypeError as e:
            return {"erreur": f"Mauvais arguments pour {nom} : {e}"}

    # ---------- outils de lecture ----------

    def rechercher_cours(self, requete, k=config.K_RECHERCHE):
        """Cherche les passages des cours ; on ne garde que ceux au-dessus du seuil de similarité."""
        resultats = self.recherche.chercher(str(requete), int(k))
        pertinents = [(m, s) for m, s in resultats if s >= self.seuil]
        if not pertinents:
            return {"passages": [], "message": "Aucun passage pertinent dans les cours."}
        return {"passages": [{"source": formater_source(m), "texte": m["texte"]} for m, s in pertinents]}

    def construire_qcm(self, sujet, nb_questions=3):
        """Fait écrire un QCM au LLM à partir des passages des cours, puis le vérifie avec du code."""
        try:
            nb = int(nb_questions)
        except (TypeError, ValueError):
            raise ErreurOutil("Le nombre de questions doit être un entier.")
        if not 1 <= nb <= 10:
            raise ErreurOutil("Le nombre de questions doit être entre 1 et 10.")

        passages = [(m, s) for m, s in self.recherche.chercher(str(sujet), 6) if s >= self.seuil]
        if not passages:
            raise ErreurOutil(f"Les cours ne parlent pas de « {sujet} », impossible de faire un QCM.")

        extraits = "\n\n".join(f"[{formater_source(m)}]\n{m['texte']}" for m, s in passages)
        messages = [
            {"role": "system", "content": PROMPT_QCM},
            {"role": "user", "content": f"Extraits :\n\n{extraits}\n\nCrée un QCM de {nb} questions sur : {sujet}"},
        ]
        rep = self.llm.chat(messages, max_tokens=1800)
        self.tokens_outils += rep["tokens"]

        questions = valider_questions(lire_liste_json(rep["contenu"]), nb)
        qcm = {
            "sujet": str(sujet),
            "questions": questions,
            "sources": sorted({formater_source(m) for m, s in passages}),
        }
        self.qcm_construits.append(qcm)
        return qcm

    def lister_sauvegardes(self):
        """Renvoie les titres des fiches et la liste des résultats déjà enregistrés."""
        return {
            "fiches": [f["titre"] for f in self.lire_fiches()],
            "resultats": self.lire_resultats(),
        }

    # ---------- outils d'écriture (confirmation obligatoire) ----------

    def enregistrer_fiche(self, titre, contenu, sources):
        """Enregistre une fiche après avoir vérifié les règles : titre, contenu assez long, sources existantes."""
        titre = str(titre).strip()
        contenu = str(contenu).strip()
        if not titre:
            raise ErreurOutil("La fiche doit avoir un titre.")
        if len(contenu) < 50:
            raise ErreurOutil("La fiche est trop courte (50 caractères minimum).")
        if isinstance(sources, str):
            sources = [sources]
        valides = [s for s in sources if self._nom_fichier(s) in self.sources_connues]
        if not valides:
            raise ErreurOutil("La fiche doit citer au moins une source des cours, par exemple 'cours_03_gan.pdf p.1'.")

        fiches = self.lire_fiches()
        fiches.append({
            "titre": titre,
            "contenu": contenu,
            "sources": valides,
            "date": datetime.now().isoformat(timespec="seconds"),
        })
        self._ecrire("fiches.json", fiches)
        return {"ok": True, "message": f"Fiche « {titre} » enregistrée."}

    def enregistrer_resultat(self, sujet, score, total):
        """Enregistre un score de QCM après avoir vérifié que 0 <= score <= total et total >= 1."""
        sujet = str(sujet).strip()
        try:
            score = int(score)
            total = int(total)
        except (TypeError, ValueError):
            raise ErreurOutil("Le score et le total doivent être des nombres entiers.")
        if not sujet:
            raise ErreurOutil("Il faut préciser le sujet du QCM.")
        if total < 1:
            raise ErreurOutil("Le total doit être au moins 1.")
        if not 0 <= score <= total:
            raise ErreurOutil(f"Score impossible : {score} sur {total}. Le score doit être entre 0 et le total.")

        resultats = self.lire_resultats()
        resultats.append({
            "sujet": sujet,
            "score": score,
            "total": total,
            "pourcentage": round(100 * score / total),
            "date": datetime.now().isoformat(timespec="seconds"),
        })
        self._ecrire("resultats.json", resultats)
        return {"ok": True, "message": f"Résultat {score}/{total} sur « {sujet} » enregistré."}

    # ---------- petites fonctions pour les fichiers ----------

    def lire_fiches(self):
        """Lit le fichier des fiches (liste vide s'il n'existe pas encore)."""
        return self._lire("fiches.json")

    def lire_resultats(self):
        """Lit le fichier des résultats (liste vide s'il n'existe pas encore)."""
        return self._lire("resultats.json")

    def _lire(self, nom):
        """Lit un fichier JSON du dossier de sauvegarde."""
        chemin = self.dossier / nom
        if not chemin.exists():
            return []
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)

    def _ecrire(self, nom, donnees):
        """Écrit un fichier JSON dans le dossier de sauvegarde."""
        self.dossier.mkdir(parents=True, exist_ok=True)
        with open(self.dossier / nom, "w", encoding="utf-8") as f:
            json.dump(donnees, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _nom_fichier(source):
        """Extrait le nom du PDF d'une source, par exemple 'cours_03_gan.pdf p.1' -> 'cours_03_gan.pdf'."""
        trouve = re.search(r"[\w\-]+\.pdf", str(source))
        return trouve.group(0) if trouve else ""
