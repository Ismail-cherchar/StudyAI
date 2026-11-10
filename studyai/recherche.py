# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/recherche.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : recherche des passages de cours par le sens
#           (embeddings + similarité cosinus)
# ------------------------------------------------------------

import hashlib
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


def formater_source(morceau):
    """Renvoie l'étiquette d'une source, par exemple 'cours_03_gan.pdf p.1'."""
    return f"{morceau['source']} p.{morceau['page']}"
