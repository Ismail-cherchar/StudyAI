# ------------------------------------------------------------
# Projet  : StudyAI - Mineure IA Générative (ECE ING5)
# Fichier : app.py
# Auteurs : Adam Bouchiba, Ismail Cherchar
# Date    : 10/11/2026
# Version : 1.0
# Rôle    : interface Streamlit (chat, QCM interactif, fiches et résultats)
# Lancer  : streamlit run app.py
# ------------------------------------------------------------

import base64
import json

import pandas as pd
import streamlit as st

from studyai import config
from studyai.assistant import NIVEAUX, StudyAI
from studyai.llm import ErreurLLM
from studyai.outils import Outils
from studyai.traces import ecrire_trace

st.set_page_config(page_title="StudyAI", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Lecture des cours et calcul des embeddings...")
def charger_systeme():
    """Charge le système une seule fois (le modèle d'embeddings est long à charger)."""
    systeme = StudyAI()
    # on charge le modèle tout de suite, sinon la première question attend ~20 s
    systeme.dense.embeddings.encoder(["préchauffage du modèle"])
    return systeme


def logo_html(largeur=120):
    """Renvoie le logo SVG du projet sous forme de balise <img> pour l'afficher dans la page."""
    with open(config.RACINE / "assets" / "logo.svg", "rb") as f:
        code = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/svg+xml;base64,{code}" width="{largeur}">'


def initialiser_session():
    """Prépare les variables gardées entre deux clics (historique, agents, QCM en cours)."""
    st.session_state.setdefault("historique", [])
    st.session_state.setdefault("agents", {})
    st.session_state.setdefault("etat_en_attente", None)
    st.session_state.setdefault("qcm", None)
    st.session_state.setdefault("correction", None)
    st.session_state.setdefault("dernier_cout", None)


def agent_du_niveau(systeme, niveau):
    """Garde un agent par niveau pour pouvoir reprendre après une confirmation."""
    if niveau not in st.session_state.agents:
        st.session_state.agents[niveau] = systeme.creer_agent(niveau)
    return st.session_state.agents[niveau]


def ajouter_reponse(resultat):
    """Ajoute la réponse dans l'historique, écrit la trace et retient le coût pour la barre latérale."""
    st.session_state.historique.append({"role": "assistant", "resultat": resultat})
    ecrire_trace(resultat)
    st.session_state.dernier_cout = {
        "niveau": resultat.get("niveau"),
        "tokens": resultat.get("tokens", 0),
        "duree": resultat.get("duree", 0.0),
    }


def afficher_reponse(resultat):
    """Affiche une réponse avec son niveau, son coût, ses sources et sa trace."""
    st.markdown(resultat.get("reponse") or "_(pas de réponse)_")
    st.caption(f"Niveau {resultat.get('niveau')} · {resultat.get('tokens', 0)} tokens · {resultat.get('duree', 0):.1f} s")
    if resultat.get("sources"):
        with st.expander("📄 Sources"):
            for s in resultat["sources"]:
                st.write("- " + s)
    if resultat.get("etapes"):
        with st.expander("🔎 Trace de l'exécution"):
            st.json(resultat["etapes"])


def barre_laterale(systeme):
    """Barre latérale : logo, choix du niveau, import de PDF et coût de la dernière réponse."""
    with st.sidebar:
        st.markdown(logo_html(90), unsafe_allow_html=True)
        st.title("StudyAI")
        page = st.radio("Menu", ["💬 Chat", "📝 QCM", "📚 Mes fiches et résultats"])
        niveau = st.radio("Niveau de l'assistant", list(NIVEAUX), format_func=lambda n: NIVEAUX[n], index=1)

        st.divider()
        st.write(f"**{len({m['source'] for m in systeme.morceaux})} cours** · {len(systeme.morceaux)} morceaux indexés")
        fichiers = st.file_uploader("Ajouter un cours (PDF)", type="pdf", accept_multiple_files=True)
        if fichiers and st.button("Importer"):
            for fichier in fichiers:
                (config.DOSSIER_COURS / fichier.name).write_bytes(fichier.getbuffer())
            st.cache_resource.clear()
            st.session_state.agents = {}
            st.rerun()

        if st.session_state.dernier_cout:
            st.divider()
            c = st.session_state.dernier_cout
            st.write("**Coût de la dernière réponse**")
            col1, col2 = st.columns(2)
            col1.metric("Tokens", c["tokens"])
            col2.metric("Latence", f"{c['duree']:.1f} s")

        if st.button("🧹 Nouvelle conversation"):
            st.session_state.historique = []
            st.session_state.etat_en_attente = None
            st.rerun()
    return page, niveau


def page_chat(systeme, niveau):
    """Page de chat : questions de cours, fiches et résultats via l'agent (avec confirmation)."""
    st.header("💬 Pose une question sur tes cours")
    for message in st.session_state.historique:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["texte"])
            else:
                afficher_reponse(message["resultat"])

    # une action d'écriture attend la validation de l'étudiant
    etat = st.session_state.etat_en_attente
    if etat:
        action = etat["action_en_attente"]
        st.warning(f"L'assistant veut lancer l'action **{action['outil']}**. Tu confirmes ?")
        st.code(json.dumps(action["arguments"], ensure_ascii=False, indent=2), language="json")
        col1, col2 = st.columns(2)
        decision = None
        if col1.button("✅ Confirmer", use_container_width=True):
            decision = True
        if col2.button("❌ Refuser", use_container_width=True):
            decision = False
        if decision is not None:
            agent = agent_du_niveau(systeme, etat["niveau"])
            try:
                etat = agent.reprendre(etat, decision)
            except ErreurLLM as e:
                st.error(str(e))
                return
            if etat["statut"] == "confirmation":
                st.session_state.etat_en_attente = etat
            else:
                st.session_state.etat_en_attente = None
                ajouter_reponse(etat)
            st.rerun()
        return

    question = st.chat_input("Ex : Qu'est-ce que l'ELBO ? / Fais-moi une fiche sur les GAN")
    if question:
        st.session_state.historique.append({"role": "user", "texte": question})
        with st.spinner("StudyAI réfléchit..."):
            try:
                agent = agent_du_niveau(systeme, niveau) if niveau >= 2 else None
                resultat = systeme.repondre(niveau, question, agent=agent)
            except ErreurLLM as e:
                st.error(f"{e}\n\nVérifie le fichier .env et que le serveur du LLM est bien lancé.")
                return
        if resultat.get("statut") == "confirmation":
            st.session_state.etat_en_attente = resultat
        else:
            ajouter_reponse(resultat)
        st.rerun()


def page_qcm(systeme, niveau):
    """Page QCM : génère un QCM depuis les cours, l'étudiant répond, on corrige et on enregistre le score."""
    st.header("📝 QCM interactif")
    outils = Outils(systeme.recherche_du_niveau(max(niveau, 2)), systeme.llm)
    col1, col2 = st.columns([3, 1])
    sujet = col1.text_input("Sujet du QCM", placeholder="Ex : les GAN, le VAE, la diffusion...")
    nb = col2.number_input("Questions", min_value=1, max_value=10, value=5)

    if st.button("Générer le QCM", type="primary") and sujet:
        with st.spinner("Création du QCM à partir de tes cours..."):
            try:
                # executer() renvoie {"erreur": ...} si une règle n'est pas respectée (ex : sujet absent des cours)
                resultat = outils.executer("construire_qcm", {"sujet": sujet, "nb_questions": int(nb)})
            except ErreurLLM as e:
                st.error(str(e))
                return
        if "erreur" in resultat:
            st.error(resultat["erreur"])
            return
        st.session_state.qcm = resultat
        st.session_state.correction = None

    qcm = st.session_state.qcm
    if not qcm:
        return
    st.subheader(f"QCM : {qcm['sujet']}")
    reponses = []
    for i, q in enumerate(qcm["questions"]):
        choix = st.radio(f"**{i + 1}. {q['question']}**", q["choix"], index=None, key=f"q{i}")
        reponses.append(None if choix is None else q["choix"].index(choix))

    if st.button("Corriger"):
        score = sum(1 for r, q in zip(reponses, qcm["questions"]) if r == q["reponse"])
        st.session_state.correction = score

    if st.session_state.correction is not None:
        score = st.session_state.correction
        total = len(qcm["questions"])
        st.success(f"Score : {score} / {total}")
        for i, q in enumerate(qcm["questions"]):
            bonne = q["choix"][q["reponse"]]
            icone = "✅" if reponses[i] == q["reponse"] else "❌"
            st.write(f"{icone} **{i + 1}.** Bonne réponse : {bonne}. {q['explication']}")
        st.caption("Sources : " + ", ".join(qcm["sources"]))
        # le clic sur ce bouton est la confirmation de l'étudiant
        if st.button("💾 Enregistrer mon résultat"):
            retour = outils.enregistrer_resultat(qcm["sujet"], score, total)
            st.toast(retour["message"])


def page_sauvegardes(systeme):
    """Page qui affiche les fiches enregistrées et l'historique des scores."""
    st.header("📚 Mes fiches et mes résultats")
    outils = Outils(systeme.dense, systeme.llm)
    fiches = outils.lire_fiches()
    resultats = outils.lire_resultats()

    st.subheader(f"Fiches ({len(fiches)})")
    if not fiches:
        st.info("Aucune fiche pour l'instant. Demande par exemple : « Fais-moi une fiche sur les GAN » (niveau 2 ou 3).")
    for fiche in fiches[::-1]:
        with st.expander(f"🗂️ {fiche['titre']} — {fiche['date'][:10]}"):
            st.markdown(fiche["contenu"])
            st.caption("Sources : " + ", ".join(fiche["sources"]))

    st.subheader(f"Résultats ({len(resultats)})")
    if resultats:
        tableau = pd.DataFrame(resultats)
        st.metric("Moyenne", f"{tableau['pourcentage'].mean():.0f} %")
        st.dataframe(tableau[["date", "sujet", "score", "total", "pourcentage"]], use_container_width=True)


def main():
    """Construit la page Streamlit."""
    initialiser_session()
    systeme = charger_systeme()
    page, niveau = barre_laterale(systeme)
    if page == "💬 Chat":
        page_chat(systeme, niveau)
    elif page == "📝 QCM":
        page_qcm(systeme, niveau)
    else:
        page_sauvegardes(systeme)


main()
