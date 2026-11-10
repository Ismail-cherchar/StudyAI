# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/assistant.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.1
# Rôle    : assemble tout le système (corpus, recherches, LLM, agents)
#           et répond avec le niveau choisi (0 à 3)
# ------------------------------------------------------------

from studyai import config
from studyai.agent import Agent
from studyai.documents import charger_corpus
from studyai.llm import LLM
from studyai.niveaux import niveau0, niveau1
from studyai.outils import Outils
from studyai.recherche import BM25, Embeddings, RechercheDense, RechercheHybride

# description courte des niveaux (affichée dans l'interface)
NIVEAUX = {
    0: "Niveau 0 - LLM seul (sans les cours)",
    1: "Niveau 1 - RAG (cherche dans les cours et cite les sources)",
    2: "Niveau 2 - Agent avec outils (fiches, QCM, résultats)",
    3: "Niveau 3 - Agent + recherche hybride BM25 + embeddings",
}


class StudyAI:
    """Regroupe tout le système et répond avec le niveau demandé."""

    def __init__(self, llm=None, embeddings=None, dossier_cours=config.DOSSIER_COURS,
                 dossier_index=config.DOSSIER_INDEX, dossier_sauvegardes=config.DOSSIER_SAUVEGARDES):
        """Charge les cours, prépare les trois moteurs de recherche et le client LLM."""
        self.llm = llm or LLM()
        self.morceaux = charger_corpus(dossier_cours)
        self.dense = RechercheDense(self.morceaux, embeddings or Embeddings(), dossier_index)
        self.bm25 = BM25(self.morceaux)
        self.hybride = RechercheHybride(self.dense, self.bm25)
        self.dossier_sauvegardes = dossier_sauvegardes

    def recherche_du_niveau(self, niveau):
        """Niveaux 1 et 2 : recherche dense. Niveau 3 : recherche hybride (notre extension)."""
        return self.hybride if niveau == 3 else self.dense

    def creer_agent(self, niveau, dossier_sauvegardes=None):
        """Crée un agent (niveau 2 ou 3) avec ses outils et son dossier de sauvegarde."""
        outils = Outils(
            self.recherche_du_niveau(niveau),
            self.llm,
            dossier_sauvegardes or self.dossier_sauvegardes,
        )
        return Agent(self.llm, outils, niveau=niveau)

    def repondre(self, niveau, question, agent=None):
        """Répond avec le niveau choisi. Aux niveaux 2 et 3, la réponse peut être une demande de confirmation."""
        if niveau == 0:
            return niveau0(question, self.llm)
        if niveau == 1:
            return niveau1(question, self.llm, self.dense)
        agent = agent or self.creer_agent(niveau)
        return agent.demarrer(question)
