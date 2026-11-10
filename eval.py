# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : eval.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : évalue les 4 niveaux sur le même jeu de test (data/tests.jsonl)
#           et produit le tableau de résultats (écrit sans framework)
#
# Utilisation :
#   python eval.py                   -> tous les niveaux (0 à 3)
#   python eval.py --niveaux 1 3     -> seulement certains niveaux
#   python eval.py --recherche       -> seulement Recall@5 et MRR (pas besoin du LLM)
#   python eval.py --essais 3        -> nombre d'essais par tâche pour pass^k
#   python eval.py --limite 3        -> 3 cas par type, pour un test rapide
# ------------------------------------------------------------

import argparse
import json
import re
import tempfile
import time
import unicodedata
from pathlib import Path

from studyai import config
from studyai.assistant import StudyAI
from studyai.llm import ErreurLLM
from studyai.traces import ecrire_trace

MARQUEURS_ABSTENTION = [
    "je ne trouve pas", "pas dans tes cours", "pas dans les cours", "ne contiennent pas",
    "ne mentionnent pas", "aucune information", "je ne sais pas", "pas d'information",
]
MARQUEURS_REFUS = MARQUEURS_ABSTENTION + [
    "hors sujet", "hors-sujet", "je ne peux pas", "pas en mesure", "je sers a", "uniquement",
    "rien a voir", "pas lie", "reviser",
]
TYPES = ["doc", "absent", "tache", "ambigu"]


# ---------------------------------------------------------------- outils de comparaison

def normaliser(texte):
    """Met en minuscules et enlève les accents pour comparer les textes plus facilement."""
    texte = unicodedata.normalize("NFD", str(texte).lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def contient_mots_cles(texte, groupes):
    """Vrai si le texte contient au moins un mot de CHAQUE groupe (ET entre groupes, OU dans un groupe)."""
    texte = normaliser(texte)
    return all(any(normaliser(mot) in texte for mot in groupe) for groupe in groupes)


def est_abstention(reponse):
    """Vrai si l'assistant dit qu'il ne trouve pas l'information au lieu d'inventer."""
    return any(m in normaliser(reponse) for m in MARQUEURS_ABSTENTION)


def gere_ambiguite(reponse, attendu):
    """Demande vague : on attend une question. Demande hors sujet : on attend un refus poli."""
    if attendu == "question":
        return "?" in reponse
    return any(m in normaliser(reponse) for m in MARQUEURS_REFUS)


def sources_citees(reponse):
    """Renvoie les noms des PDF cités dans la réponse."""
    return set(re.findall(r"[\w\-]+\.pdf", reponse))


def est_fidele(resultat):
    """Vrai si la réponse cite au moins une source et que toutes les sources citées ont vraiment été lues."""
    citees = sources_citees(resultat.get("reponse", ""))
    lues = {s.split(" p.")[0] for s in resultat.get("sources", [])}
    return bool(citees) and citees <= lues


def lire_json(chemin):
    """Lit un fichier JSON de sauvegarde (liste vide s'il n'existe pas)."""
    chemin = Path(chemin)
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def verifier_etat(verif, dossier, resultat):
    """Vérifie par du code l'état final après une tâche (fichiers enregistrés et QCM construits)."""
    fiches = lire_json(Path(dossier) / "fiches.json")
    resultats = lire_json(Path(dossier) / "resultats.json")
    qcms = resultat.get("qcm_construits", [])

    if verif.get("aucune_sauvegarde") and (fiches or resultats):
        return False
    if "fiche" in verif:
        groupes = verif["fiche"]["contient"]
        if not any(contient_mots_cles(f["titre"] + " " + f["contenu"], groupes) and f["sources"] for f in fiches):
            return False
    if "resultat" in verif:
        v = verif["resultat"]
        if not any(
            contient_mots_cles(r["sujet"], v["sujet"]) and r["score"] == v["score"] and r["total"] == v["total"]
            for r in resultats
        ):
            return False
    if "qcm" in verif:
        if not any(len(q["questions"]) == verif["qcm"]["nb_questions"] for q in qcms):
            return False
    return True


# ---------------------------------------------------------------- recherche : Recall@k et MRR

def est_pertinent(morceau, test):
    """Un morceau est pertinent s'il vient du bon cours ET contient le passage annoté qui répond à la question."""
    return morceau["source"] == test["source"] and normaliser(test["passage"]) in normaliser(morceau["texte"])


def rang_premier_pertinent(resultats, test):
    """Renvoie le rang (1, 2, ...) du premier morceau pertinent dans les résultats, ou None."""
    for rang, (morceau, score) in enumerate(resultats, start=1):
        if est_pertinent(morceau, test):
            return rang
    return None


def evaluer_recherche(moteur, tests, k=5):
    """Calcule Recall@k et MRR sur les questions dont la réponse est dans les cours."""
    questions = [t for t in tests if t["type"] == "doc"]
    trouves, somme_rr = 0, 0.0
    for t in questions:
        rang = rang_premier_pertinent(moteur.chercher(t["question"], k), t)
        if rang is not None:
            trouves += 1
            somme_rr += 1 / rang
    return trouves / len(questions), somme_rr / len(questions)


# ---------------------------------------------------------------- exécution d'un cas

def executer(systeme, niveau, question, dossier):
    """Lance une question. Pour les agents, l'étudiant simulé confirme les actions demandées."""
    if niveau in (0, 1):
        return systeme.repondre(niveau, question)
    agent = systeme.creer_agent(niveau, dossier)
    etat = agent.demarrer(question)
    confirmations = 0
    while etat["statut"] == "confirmation" and confirmations < 5:
        etat = agent.reprendre(etat, accepte=True)
        confirmations += 1
    etat["qcm_construits"] = agent.outils.qcm_construits
    return etat


def juger(test, resultat, niveau, dossier):
    """Renvoie un dictionnaire de critères réussis (True/False) pour ce cas de test."""
    reponse = resultat.get("reponse", "")
    if test["type"] == "doc":
        jugement = {"exact": contient_mots_cles(reponse, test["mots_cles"])}
        if niveau >= 1:
            jugement["fidele"] = est_fidele(resultat)
        return jugement
    if test["type"] == "absent":
        return {"abstention": est_abstention(reponse)}
    if test["type"] == "tache":
        return {"succes": verifier_etat(test["verif"], dossier, resultat)}
    # demande ambiguë ou hors sujet : en plus, un agent ne doit rien enregistrer
    ok = gere_ambiguite(reponse, test["attendu"])
    if niveau >= 2 and "verif" in test:
        ok = ok and verifier_etat(test["verif"], dossier, resultat)
    return {"ambigu": ok}


def moyenne(valeurs):
    """Moyenne d'une liste (None si la liste est vide)."""
    return sum(valeurs) / len(valeurs) if valeurs else None


def evaluer_niveau(systeme, niveau, tests, essais):
    """Passe tous les cas de test sur un niveau et renvoie les métriques + les lignes brutes."""
    lignes, tokens, latences = [], [], []
    criteres = {"exact": [], "fidele": [], "abstention": [], "ambigu": [], "succes": [], "pass_k": []}

    for test in tests:
        if test["type"] == "tache" and niveau < 2:
            continue  # les niveaux 0 et 1 n'ont pas d'outils, ils ne peuvent pas faire les tâches
        nb_essais = essais if test["type"] == "tache" else 1
        reussites = []
        for essai in range(nb_essais):
            with tempfile.TemporaryDirectory() as dossier:
                debut = time.time()
                try:
                    resultat = executer(systeme, niveau, test["question"], dossier)
                except ErreurLLM as e:
                    resultat = {"niveau": niveau, "question": test["question"], "reponse": "",
                                "erreur": str(e), "tokens": 0, "duree": 0.0, "sources": [], "etapes": []}
                latence = time.time() - debut
                jugement = juger(test, resultat, niveau, dossier)

            tokens.append(resultat.get("tokens", 0))
            latences.append(latence)
            ecrire_trace(resultat, config.RACINE / "traces" / f"eval_niveau_{niveau}.jsonl")
            lignes.append({
                "id": test["id"], "type": test["type"], "essai": essai + 1, "niveau": niveau,
                "question": test["question"], "reponse": resultat.get("reponse", ""),
                "statut": resultat.get("statut", "termine"), "sources": resultat.get("sources", []),
                "jugement": jugement, "tokens": resultat.get("tokens", 0), "latence": round(latence, 2),
                "erreur": resultat.get("erreur"),
            })

            if test["type"] == "tache":
                reussites.append(jugement["succes"])
            else:
                for cle, valeur in jugement.items():
                    criteres[cle].append(valeur)
        if test["type"] == "tache":
            criteres["succes"].append(reussites[0])
            criteres["pass_k"].append(all(reussites))
        print(f"  niveau {niveau} | {test['id']} | {jugement}")

    metriques = {cle: moyenne(v) for cle, v in criteres.items()}
    metriques["tokens"] = moyenne(tokens)
    metriques["latence"] = moyenne(latences)
    return metriques, lignes


# ---------------------------------------------------------------- affichage

def pourcent(x):
    """Affiche un nombre entre 0 et 1 en pourcentage, ou un tiret s'il n'y a pas de valeur."""
    return "—" if x is None else f"{100 * x:.0f} %"


def tableau_markdown(lignes_tableau, essais):
    """Construit le tableau final des niveaux au format Markdown."""
    entete = ("| Niveau | Recall@5 | MRR | Exactitude | Fidélité | Abstention | Précision / refus "
              f"| Succès tâches | pass^{essais} | Tokens / requête | Latence (s) |")
    sep = "|" + "---|" * 11
    lignes = [entete, sep]
    for niveau, m in lignes_tableau:
        mrr = "—" if m.get("mrr") is None else f"{m['mrr']:.2f}"
        tok = "—" if m.get("tokens") is None else f"{m['tokens']:.0f}"
        lat = "—" if m.get("latence") is None else f"{m['latence']:.1f}"
        lignes.append(
            f"| {niveau} | {pourcent(m.get('recall'))} | {mrr} | {pourcent(m.get('exact'))} "
            f"| {pourcent(m.get('fidele'))} | {pourcent(m.get('abstention'))} | {pourcent(m.get('ambigu'))} "
            f"| {pourcent(m.get('succes'))} | {pourcent(m.get('pass_k'))} | {tok} | {lat} |"
        )
    return "\n".join(lignes)


def charger_tests(chemin=config.FICHIER_TESTS, limite=None):
    """Lit le fichier tests.jsonl ; avec `limite`, on garde seulement les premiers cas de chaque type."""
    with open(chemin, encoding="utf-8") as f:
        tests = [json.loads(l) for l in f if l.strip()]
    if limite:
        tests = [t for typ in TYPES for t in [x for x in tests if x["type"] == typ][:limite]]
    return tests


def main():
    """Point d'entrée : lit les options, lance l'évaluation et enregistre les résultats."""
    parser = argparse.ArgumentParser(description="Évaluation des niveaux de StudyAI")
    parser.add_argument("--niveaux", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--essais", type=int, default=3, help="essais par tâche pour pass^k")
    parser.add_argument("--recherche", action="store_true", help="évaluer seulement la recherche (sans LLM)")
    parser.add_argument("--limite", type=int, default=None, help="nombre max de cas par type")
    args = parser.parse_args()

    tests = charger_tests(limite=args.limite)
    print(f"{len(tests)} cas de test chargés")
    systeme = StudyAI()
    print(f"{len(systeme.morceaux)} morceaux de cours indexés")
    config.DOSSIER_RESULTATS.mkdir(exist_ok=True)

    # 1) la recherche seule : pas besoin du LLM
    lignes_recherche = ["| Moteur | Recall@5 | MRR |", "|---|---|---|"]
    for moteur in (systeme.bm25, systeme.dense, systeme.hybride):
        recall, mrr = evaluer_recherche(moteur, tests)
        lignes_recherche.append(f"| {moteur.nom} | {pourcent(recall)} | {mrr:.2f} |")
    texte_recherche = "\n".join(lignes_recherche)
    print("\n" + texte_recherche + "\n")
    (config.DOSSIER_RESULTATS / "recherche.md").write_text(texte_recherche + "\n", encoding="utf-8")
    if args.recherche:
        return

    # 2) chaque niveau sur le même protocole
    lignes_tableau = []
    for niveau in args.niveaux:
        print(f"--- Niveau {niveau} ---")
        metriques, lignes = evaluer_niveau(systeme, niveau, tests, args.essais)
        if niveau >= 1:
            metriques["recall"], metriques["mrr"] = evaluer_recherche(systeme.recherche_du_niveau(niveau), tests)
        lignes_tableau.append((niveau, metriques))
        with open(config.DOSSIER_RESULTATS / f"niveau_{niveau}.jsonl", "w", encoding="utf-8") as f:
            for l in lignes:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")

    tableau = tableau_markdown(lignes_tableau, args.essais)
    print("\n" + tableau)
    (config.DOSSIER_RESULTATS / "tableau.md").write_text(tableau + "\n", encoding="utf-8")
    print(f"\nRésultats enregistrés dans {config.DOSSIER_RESULTATS}")


if __name__ == "__main__":
    main()
