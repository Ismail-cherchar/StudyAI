# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : studyai/documents.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : lit les cours PDF et les découpe en petits morceaux (chunks)
# ------------------------------------------------------------

import re
from pathlib import Path

from pypdf import PdfReader

from studyai import config


def nettoyer(texte):
    """Recolle les mots coupés en fin de ligne et enlève les espaces en trop."""
    texte = re.sub(r"-\n(\w)", r"\1", texte)
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def lire_pdf(chemin):
    """Lit un PDF et renvoie la liste des (numéro de page, texte de la page)."""
    lecteur = PdfReader(str(chemin))
    pages = []
    for numero, page in enumerate(lecteur.pages, start=1):
        texte = nettoyer(page.extract_text() or "")
        if texte:
            pages.append((numero, texte))
    return pages


def decouper(texte, taille=config.TAILLE_MORCEAU, chevauchement=config.CHEVAUCHEMENT):
    """Découpe un texte en morceaux de `taille` mots qui se chevauchent de `chevauchement` mots."""
    mots = texte.split()
    if len(mots) <= taille:
        return [texte]
    morceaux = []
    pas = taille - chevauchement
    for debut in range(0, len(mots), pas):
        morceau = mots[debut:debut + taille]
        morceaux.append(" ".join(morceau))
        if debut + taille >= len(mots):
            break
    return morceaux


def charger_corpus(dossier=config.DOSSIER_COURS):
    """Lit tous les PDF d'un dossier et renvoie les morceaux avec leur source et leur page."""
    morceaux = []
    for chemin in sorted(Path(dossier).glob("*.pdf")):
        for numero_page, texte in lire_pdf(chemin):
            for i, bout in enumerate(decouper(texte)):
                morceaux.append({
                    "id": f"{chemin.name}#p{numero_page}#{i}",
                    "source": chemin.name,
                    "page": numero_page,
                    "texte": bout,
                })
    return morceaux
