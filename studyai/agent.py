# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/agent.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : boucle de l'agent écrite à la main (sans framework agentique)
#           raisonner -> appeler un outil -> observer -> recommencer
#           boucle bornée + confirmation avant toute écriture
# ------------------------------------------------------------

import json
import time

from studyai import config
from studyai.niveaux import PHRASE_ABSTENTION

PROMPT_AGENT = f"""Tu es StudyAI, un assistant de révision qui peut utiliser des outils.
Règles :
- Pour une question de cours, appelle d'abord rechercher_cours, puis réponds uniquement avec les passages trouvés en citant la source entre crochets, par exemple [cours_03_gan.pdf p.1].
- Si la recherche ne trouve rien d'utile, réponds exactement : "{PHRASE_ABSTENTION}"
- Pour une fiche de révision : appelle rechercher_cours, rédige la fiche (notions clés, définitions, un exemple), puis appelle enregistrer_fiche avec les sources utilisées.
- Pour un QCM : appelle construire_qcm.
- Pour enregistrer un score : appelle enregistrer_resultat avec le sujet, le score et le total donnés par l'étudiant.
- Si la demande est trop vague (sujet ou score manquant), pose une question pour la préciser au lieu d'agir.
- Si la demande n'a rien à voir avec les révisions, refuse poliment sans utiliser d'outil.
- Si un outil renvoie une erreur, explique-la à l'étudiant ou corrige ton appel.
- Réponds en français, de façon courte."""


def lire_arguments(arguments):
    """Les arguments d'un appel d'outil arrivent en texte JSON : on les transforme en dictionnaire."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    resultat = json.loads(arguments)
    if not isinstance(resultat, dict):
        raise json.JSONDecodeError("les arguments doivent être un objet JSON", str(arguments), 0)
    return resultat


def resumer(resultat, longueur=400):
    """Raccourcit le résultat d'un outil pour que la trace reste lisible."""
    texte = json.dumps(resultat, ensure_ascii=False)
    return texte if len(texte) <= longueur else texte[:longueur] + "..."


class Agent:
    """Agent outillé : boucle ReAct écrite à la main, bornée en nombre d'étapes et en tokens."""

    def __init__(self, llm, outils, niveau=2, max_etapes=config.MAX_ETAPES_AGENT,
                 max_tokens=config.MAX_TOKENS_AGENT):
        """Garde le LLM, les outils et les limites de la boucle."""
        self.llm = llm
        self.outils = outils
        self.niveau = niveau
        self.max_etapes = max_etapes
        self.max_tokens = max_tokens

    def demarrer(self, question):
        """Crée l'état de départ de la conversation puis lance la boucle."""
        etat = {
            "niveau": self.niveau,
            "question": question,
            "messages": [
                {"role": "system", "content": PROMPT_AGENT},
                {"role": "user", "content": question},
            ],
            "appels": [],             # appels d'outils pas encore exécutés
            "statut": "en_cours",     # en_cours / confirmation / termine / limite
            "action_en_attente": None,
            "reponse": "",
            "sources": [],
            "etapes": [],
            "tokens": 0,
            "duree": 0.0,
            "nb_appels_llm": 0,
        }
        return self._boucle(etat)

    def reprendre(self, etat, accepte):
        """Reprend la boucle après la réponse de l'utilisateur à une demande de confirmation."""
        if etat["statut"] != "confirmation":
            return etat
        appel = etat["appels"][0]
        appel["decision"] = "accepte" if accepte else "refuse"
        etat["etapes"].append({"type": "confirmation", "outil": appel["nom"], "accepte": accepte})
        etat["statut"] = "en_cours"
        etat["action_en_attente"] = None
        return self._boucle(etat)

    def _boucle(self, etat):
        """Boucle principale : on demande au LLM quoi faire jusqu'à une réponse finale ou une limite."""
        if not self._traiter_appels(etat):
            return etat

        while etat["nb_appels_llm"] < self.max_etapes:
            if etat["tokens"] >= self.max_tokens:
                etat["reponse"] = "J'ai dépassé le budget de tokens prévu pour cette demande, je m'arrête ici."
                etat["statut"] = "limite"
                return etat

            rep = self.llm.chat(etat["messages"], outils=self.outils.schemas())
            etat["nb_appels_llm"] += 1
            etat["tokens"] += rep["tokens"]
            etat["duree"] += rep["duree"]
            noms = [a.get("function", {}).get("name") for a in rep["appels"]]
            etat["etapes"].append({"type": "llm", "tokens": rep["tokens"], "duree": rep["duree"], "outils_demandes": noms})

            # pas d'appel d'outil : c'est la réponse finale
            if not rep["appels"]:
                etat["reponse"] = rep["contenu"]
                etat["statut"] = "termine"
                return etat

            # on recopie les appels avec un identifiant (certains modèles n'en donnent pas)
            appels_propres = []
            for n, brut in enumerate(rep["appels"]):
                fonction = brut.get("function", {})
                appel = {"id": brut.get("id") or f"appel_{etat['nb_appels_llm']}_{n}", "nom": fonction.get("name", "")}
                try:
                    appel["arguments"] = lire_arguments(fonction.get("arguments"))
                except json.JSONDecodeError:
                    appel["arguments"] = {}
                    appel["erreur"] = "Les arguments ne sont pas un JSON valide, recommence l'appel."
                etat["appels"].append(appel)
                arguments_texte = fonction.get("arguments")
                if not isinstance(arguments_texte, str):
                    arguments_texte = json.dumps(arguments_texte or {}, ensure_ascii=False)
                appels_propres.append({
                    "id": appel["id"], "type": "function",
                    "function": {"name": appel["nom"], "arguments": arguments_texte},
                })
            etat["messages"].append({"role": "assistant", "content": rep["contenu"], "tool_calls": appels_propres})

            if not self._traiter_appels(etat):
                return etat

        etat["reponse"] = "J'ai atteint le nombre maximum d'étapes sans finir. Peux-tu reformuler ta demande plus simplement ?"
        etat["statut"] = "limite"
        return etat

    def _traiter_appels(self, etat):
        """Exécute les appels d'outils en attente. Renvoie False si on doit attendre une confirmation."""
        while etat["appels"]:
            appel = etat["appels"][0]

            # action d'écriture pas encore validée : on met la boucle en pause
            if self.outils.est_sensible(appel["nom"]) and "decision" not in appel and "erreur" not in appel:
                etat["statut"] = "confirmation"
                etat["action_en_attente"] = {"outil": appel["nom"], "arguments": appel["arguments"]}
                return False

            etat["appels"].pop(0)
            debut = time.time()
            tokens_avant = self.outils.tokens_outils
            if "erreur" in appel:
                resultat = {"erreur": appel["erreur"]}
            elif appel.get("decision") == "refuse":
                resultat = {"erreur": "Action refusée par l'utilisateur. Ne la refais pas sans nouvelle demande."}
            else:
                resultat = self.outils.executer(appel["nom"], appel["arguments"])
            duree = time.time() - debut
            tokens_outil = self.outils.tokens_outils - tokens_avant

            etat["tokens"] += tokens_outil
            etat["duree"] += duree
            self._noter_sources(etat, resultat)
            etat["messages"].append({
                "role": "tool",
                "tool_call_id": appel["id"],
                "content": json.dumps(resultat, ensure_ascii=False),
            })
            etat["etapes"].append({
                "type": "outil",
                "nom": appel["nom"],
                "arguments": appel["arguments"],
                "resultat": resumer(resultat),
                "tokens": tokens_outil,
                "duree": round(duree, 3),
            })
        return True

    @staticmethod
    def _noter_sources(etat, resultat):
        """Garde la liste des sources que l'agent a vraiment lues (pour vérifier ses citations)."""
        if not isinstance(resultat, dict):
            return
        sources = [p["source"] for p in resultat.get("passages", [])] + list(resultat.get("sources", []))
        for s in sources:
            if s not in etat["sources"]:
                etat["sources"].append(s)
