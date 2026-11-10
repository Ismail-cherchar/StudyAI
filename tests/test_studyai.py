# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : tests/test_studyai.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : tests unitaires (sans vrai LLM ni vrai modèle d'embeddings)
# Lancer  : pytest
# ------------------------------------------------------------

import json

import numpy as np

import eval as evaluation
from studyai.agent import Agent
from studyai.documents import decouper
from studyai.niveaux import PHRASE_ABSTENTION, niveau1
from studyai.outils import Outils
from studyai.recherche import BM25, RechercheDense, RechercheHybride, tokeniser


class FauxEmbeddings:
    """Faux modèle d'embeddings : un sac de mots haché dans 64 cases (suffisant pour les tests)."""

    nom_modele = "faux"

    def encoder(self, textes):
        """Renvoie un vecteur normalisé par texte."""
        vecteurs = np.zeros((len(textes), 64), dtype=np.float32)
        for i, texte in enumerate(textes):
            for mot in tokeniser(texte):
                vecteurs[i, sum(ord(c) for c in mot) % 64] += 1
        normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
        return vecteurs / np.maximum(normes, 1e-9)


class FauxLLM:
    """Faux LLM qui renvoie des réponses écrites à l'avance, dans l'ordre."""

    def __init__(self, reponses):
        """Garde la liste des réponses à renvoyer."""
        self.reponses = list(reponses)
        self.nb_appels = 0

    def chat(self, messages, outils=None, max_tokens=800):
        """Renvoie la réponse suivante (la dernière est répétée si la liste est finie)."""
        self.nb_appels += 1
        rep = self.reponses.pop(0) if len(self.reponses) > 1 else self.reponses[0]
        return {"contenu": rep.get("contenu", ""), "appels": rep.get("appels", []), "tokens": 10, "duree": 0.01}


def appel_outil(nom, arguments, identifiant="a1"):
    """Fabrique un appel d'outil au format de l'API OpenAI."""
    return {"id": identifiant, "type": "function",
            "function": {"name": nom, "arguments": json.dumps(arguments)}}


MORCEAUX = [
    {"id": "gan#1", "source": "cours_03_gan.pdf", "page": 1,
     "texte": "Le générateur et le discriminateur d'un GAN sont entraînés ensemble, proposé par Goodfellow en 2014."},
    {"id": "vae#1", "source": "cours_02_vae.pdf", "page": 1,
     "texte": "Le VAE maximise l'ELBO : terme de reconstruction et divergence KL."},
    {"id": "dif#1", "source": "cours_04_diffusion.pdf", "page": 1,
     "texte": "DDIM échantillonne en 50 pas au lieu des 1000 pas de DDPM avec un U-Net."},
]


def creer_outils(tmp_path, llm=None):
    """Crée des outils avec la fausse recherche et un dossier de sauvegarde temporaire."""
    dense = RechercheDense(MORCEAUX, FauxEmbeddings(), dossier_cache=None)
    return Outils(dense, llm or FauxLLM([{"contenu": "ok"}]), tmp_path, seuil=0.1)


# ---------------------------------------------------------------- découpage et recherche

def test_decouper_chevauchement():
    """Les morceaux doivent se chevaucher du bon nombre de mots."""
    texte = " ".join(str(i) for i in range(100))
    morceaux = decouper(texte, taille=40, chevauchement=10)
    assert morceaux[0].split()[-10:] == morceaux[1].split()[:10]
    assert morceaux[-1].split()[-1] == "99"


def test_bm25_trouve_le_bon_morceau():
    """BM25 doit classer en premier le morceau qui contient les mots de la requête."""
    resultats = BM25(MORCEAUX).chercher("discriminateur Goodfellow", k=3)
    assert resultats[0][0]["id"] == "gan#1"


def test_hybride_renvoie_k_resultats():
    """La recherche hybride renvoie k morceaux, le plus pertinent en premier."""
    dense = RechercheDense(MORCEAUX, FauxEmbeddings(), dossier_cache=None)
    hybride = RechercheHybride(dense, BM25(MORCEAUX))
    resultats = hybride.chercher("ELBO reconstruction VAE", k=2)
    assert len(resultats) == 2
    assert resultats[0][0]["id"] == "vae#1"


def test_niveau1_abstention_sans_appel_llm():
    """Si aucun passage n'est proche, le niveau 1 s'abstient sans appeler le LLM."""
    dense = RechercheDense(MORCEAUX, FauxEmbeddings(), dossier_cache=None)
    llm = FauxLLM([{"contenu": "réponse inventée"}])
    resultat = niveau1("recette pâte crêpes", llm, dense, seuil=0.35)
    assert resultat["reponse"] == PHRASE_ABSTENTION
    assert llm.nb_appels == 0


# ---------------------------------------------------------------- règles métier des outils

def test_score_superieur_au_total_refuse(tmp_path):
    """Un score de 12/10 est refusé et rien n'est écrit."""
    outils = creer_outils(tmp_path)
    retour = outils.executer("enregistrer_resultat", {"sujet": "LLM", "score": 12, "total": 10})
    assert "erreur" in retour
    assert outils.lire_resultats() == []


def test_fiche_sans_source_refusee(tmp_path):
    """Une fiche qui ne cite aucun cours existant est refusée."""
    outils = creer_outils(tmp_path)
    retour = outils.executer("enregistrer_fiche", {"titre": "GAN", "contenu": "x" * 60, "sources": ["wikipedia"]})
    assert "erreur" in retour


def test_qcm_valide_et_invalide(tmp_path):
    """Un QCM avec 4 choix est accepté, un QCM avec 3 choix est refusé."""
    bon = [{"question": "Qui a proposé les GAN ?", "choix": ["Goodfellow", "Kingma", "Ho", "Vaswani"],
            "reponse": 0, "explication": "Goodfellow en 2014."}]
    outils = creer_outils(tmp_path, FauxLLM([{"contenu": "Voici : " + json.dumps(bon)}]))
    assert len(outils.executer("construire_qcm", {"sujet": "GAN", "nb_questions": 1})["questions"]) == 1

    mauvais = [{"question": "Q ?", "choix": ["a", "b", "c"], "reponse": 0}]
    outils = creer_outils(tmp_path, FauxLLM([{"contenu": json.dumps(mauvais)}]))
    assert "erreur" in outils.executer("construire_qcm", {"sujet": "GAN", "nb_questions": 1})


# ---------------------------------------------------------------- boucle de l'agent

def test_agent_attend_la_confirmation(tmp_path):
    """L'agent se met en pause avant d'écrire ; refuser n'écrit rien, accepter écrit le résultat."""
    scenario = [
        {"appels": [appel_outil("enregistrer_resultat", {"sujet": "GAN", "score": 4, "total": 5})]},
        {"contenu": "C'est fait."},
    ]
    # 1) l'étudiant refuse
    outils = creer_outils(tmp_path)
    agent = Agent(FauxLLM(scenario), outils)
    etat = agent.demarrer("J'ai eu 4/5 sur les GAN, enregistre-le")
    assert etat["statut"] == "confirmation"
    assert outils.lire_resultats() == []
    etat = agent.reprendre(etat, accepte=False)
    assert etat["statut"] == "termine"
    assert outils.lire_resultats() == []

    # 2) l'étudiant accepte
    agent = Agent(FauxLLM(scenario), outils)
    etat = agent.reprendre(agent.demarrer("J'ai eu 4/5 sur les GAN, enregistre-le"), accepte=True)
    assert etat["statut"] == "termine"
    assert outils.lire_resultats()[0]["score"] == 4


def test_agent_boucle_bornee(tmp_path):
    """Un modèle qui appelle des outils à l'infini est arrêté par la limite d'étapes."""
    llm = FauxLLM([{"appels": [appel_outil("rechercher_cours", {"requete": "GAN"})]}])
    agent = Agent(llm, creer_outils(tmp_path), max_etapes=3)
    etat = agent.demarrer("question")
    assert etat["statut"] == "limite"
    assert llm.nb_appels == 3


def test_agent_erreur_outil_renvoyee_au_modele(tmp_path):
    """Une erreur d'outil ne fait pas planter l'agent : elle est renvoyée au modèle."""
    scenario = [
        {"appels": [appel_outil("outil_qui_existe_pas", {})]},
        {"contenu": "Désolé, je ne peux pas faire ça."},
    ]
    etat = Agent(FauxLLM(scenario), creer_outils(tmp_path)).demarrer("test")
    assert etat["statut"] == "termine"
    assert "Outil inconnu" in etat["messages"][-1]["content"]


# ---------------------------------------------------------------- fonctions de l'évaluation

def test_mots_cles_et_abstention():
    """Vérifie les règles utilisées par eval.py pour juger les réponses."""
    assert evaluation.contient_mots_cles("Proposé par Ian Goodfellow en 2014", [["goodfellow"], ["2014"]])
    assert not evaluation.contient_mots_cles("Proposé en 2014", [["goodfellow"], ["2014"]])
    assert evaluation.est_abstention(PHRASE_ABSTENTION)
    assert evaluation.gere_ambiguite("Sur quel sujet veux-tu la fiche ?", "question")


def test_fidelite_aux_sources():
    """Une réponse est fidèle si elle cite seulement des sources réellement lues."""
    resultat = {"reponse": "Le FID doit être bas [cours_03_gan.pdf p.2].", "sources": ["cours_03_gan.pdf p.2"]}
    assert evaluation.est_fidele(resultat)
    resultat["reponse"] = "Le FID doit être bas [cours_99_invente.pdf p.1]."
    assert not evaluation.est_fidele(resultat)
