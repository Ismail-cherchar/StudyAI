# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/recherche.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.1
# Rôle    : recherche des passages de cours
#           - dense (embeddings + similarité cosinus)  -> niveaux 1 et 2
#           - BM25 écrit à la main (mots-clés)
#           - hybride dense + BM25 avec fusion RRF     -> niveau 3
# ------------------------------------------------------------

import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from studyai import config


class Embeddings:
    """Transforme des textes en vecteurs avec un modèle sentence-transformers (chargé une seule fois)."""

    def __init__(self, nom_modele=config.MODELE_EMBEDDINGS):
        """Garde le nom du modèle, il sera chargé au premier appel (c'est long)."""
        self.nom_modele = nom_modele
        self.modele = None

    def encoder(self, textes):
        """Renvoie une matrice (nb_textes x dimension) de vecteurs normalisés."""
        if self.modele is None:
            from sentence_transformers import SentenceTransformer
            self.modele = SentenceTransformer(self.nom_modele)
        vecteurs = self.modele.encode(textes, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecteurs, dtype=np.float32)


class RechercheDense:
    """Recherche par sens : on compare le vecteur de la question à celui de chaque morceau."""

    nom = "dense"

    def __init__(self, morceaux, embeddings, dossier_cache=config.DOSSIER_INDEX):
        """Calcule (ou recharge depuis le cache) les vecteurs de tous les morceaux."""
        self.morceaux = morceaux
        self.embeddings = embeddings
        self.vecteurs = self._charger_ou_calculer(dossier_cache)

    def _charger_ou_calculer(self, dossier_cache):
        """Évite de recalculer les vecteurs à chaque lancement : on les garde dans un fichier .npy."""
        if not self.morceaux:
            return np.zeros((0, 1), dtype=np.float32)
        textes = [m["texte"] for m in self.morceaux]
        nom_modele = getattr(self.embeddings, "nom_modele", "inconnu")
        empreinte = hashlib.md5((nom_modele + "".join(textes)).encode("utf-8")).hexdigest()[:12]
        if dossier_cache is not None:
            fichier = Path(dossier_cache) / f"vecteurs_{empreinte}.npy"
            if fichier.exists():
                return np.load(fichier)
        vecteurs = self.embeddings.encoder(textes)
        if dossier_cache is not None:
            Path(dossier_cache).mkdir(parents=True, exist_ok=True)
            np.save(fichier, vecteurs)
        return vecteurs

    def similarites(self, requete):
        """Renvoie la similarité cosinus entre la requête et chaque morceau."""
        if not self.morceaux:
            return np.zeros(0)
        vecteur = self.embeddings.encoder([requete])[0]
        return self.vecteurs @ vecteur

    def chercher(self, requete, k=config.K_RECHERCHE):
        """Renvoie les k morceaux les plus proches sous forme de liste (morceau, score)."""
        scores = self.similarites(requete)
        ordre = np.argsort(-scores)[:k]
        return [(self.morceaux[i], float(scores[i])) for i in ordre]


MOTS_VIDES = set("""
le la les un une des du de d l au aux et ou en a est sont que qui quoi quel quelle quels quelles
ce cet cette ces se sa son ses sur par pour dans avec sans pas ne plus il elle on y
je tu nous vous ils elles me te moi toi mon ma mes ton ta tes leur leurs comment pourquoi
quand combien donne donner explique expliquer c qu s n t
""".split())


def tokeniser(texte):
    """Met le texte en minuscules, enlève les accents et garde les mots utiles (pour BM25)."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    mots = re.findall(r"[a-z0-9]+", texte)
    return [m for m in mots if m not in MOTS_VIDES and len(m) > 1]


class BM25:
    """Recherche par mots-clés avec la formule BM25 (écrite à la main, sans librairie)."""

    nom = "bm25"

    def __init__(self, morceaux, k1=1.5, b=0.75):
        """Prépare les statistiques : fréquence des mots par morceau, longueur moyenne et IDF."""
        self.morceaux = morceaux
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokeniser(m["texte"])) for m in morceaux]
        self.longueurs = [sum(d.values()) for d in self.docs]
        self.longueur_moyenne = (sum(self.longueurs) / len(self.longueurs)) if self.longueurs else 0
        # nombre de morceaux qui contiennent chaque mot
        nb_docs_avec_mot = Counter()
        for d in self.docs:
            nb_docs_avec_mot.update(d.keys())
        n = len(self.docs)
        self.idf = {
            mot: math.log(1 + (n - nb + 0.5) / (nb + 0.5)) for mot, nb in nb_docs_avec_mot.items()
        }

    def scores(self, requete):
        """Calcule le score BM25 de la requête pour chaque morceau."""
        mots = tokeniser(requete)
        resultats = []
        for doc, longueur in zip(self.docs, self.longueurs):
            score = 0.0
            for mot in mots:
                if mot not in doc:
                    continue
                f = doc[mot]
                normalisation = 1 - self.b + self.b * longueur / self.longueur_moyenne
                score += self.idf[mot] * f * (self.k1 + 1) / (f + self.k1 * normalisation)
            resultats.append(score)
        return np.array(resultats)

    def chercher(self, requete, k=config.K_RECHERCHE):
        """Renvoie les k morceaux avec le meilleur score BM25."""
        scores = self.scores(requete)
        ordre = np.argsort(-scores)[:k]
        return [(self.morceaux[i], float(scores[i])) for i in ordre]


class RechercheHybride:
    """Combine la recherche dense et BM25 avec la fusion RRF (Reciprocal Rank Fusion)."""

    nom = "hybride"

    def __init__(self, dense, bm25, k_rrf=60):
        """Garde les deux moteurs de recherche et la constante k de la formule RRF."""
        self.dense = dense
        self.bm25 = bm25
        self.morceaux = dense.morceaux
        self.k_rrf = k_rrf

    def chercher(self, requete, k=config.K_RECHERCHE):
        """Fusionne les deux classements : score = somme de 1 / (k_rrf + rang).

        Le score renvoyé est la similarité dense, pour pouvoir garder le même seuil d'abstention.
        """
        if not self.morceaux:
            return []
        sim_dense = self.dense.similarites(requete)
        score_bm25 = self.bm25.scores(requete)
        rang_dense = np.argsort(-sim_dense)
        rang_bm25 = np.argsort(-score_bm25)

        fusion = np.zeros(len(self.morceaux))
        for rang, i in enumerate(rang_dense):
            fusion[i] += 1 / (self.k_rrf + rang + 1)
        for rang, i in enumerate(rang_bm25):
            if score_bm25[i] > 0:  # un morceau sans aucun mot commun n'a pas de rang BM25
                fusion[i] += 1 / (self.k_rrf + rang + 1)

        ordre = np.argsort(-fusion)[:k]
        return [(self.morceaux[i], float(sim_dense[i])) for i in ordre]


def formater_source(morceau):
    """Renvoie l'étiquette d'une source, par exemple 'cours_03_gan.pdf p.1'."""
    return f"{morceau['source']} p.{morceau['page']}"
