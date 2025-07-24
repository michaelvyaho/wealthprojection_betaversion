import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from math import sqrt
from scipy.stats import norm
import numpy_financial as npf
from modules.Utils import *
from modules.pdf_generator import *
from annual_return_index import *


# CONFIGURATION
st.set_page_config(page_title="Simulateur Patrimoine", layout="wide")
st.title("💰 Simulateur de Valorisation du Patrimoine")

# ------------------ PROFIL ------------------
st.header("👤 Profil Utilisateur")
birth_year = st.number_input("Votre année de naissance", min_value=1925, max_value=datetime.now().year, value=1986)


current_year = datetime.now().year
current_age = current_year - birth_year
st.write(f"Âge actuel : **{current_age} ans**")
start_year = st.number_input(f"Année du premier investissement", min_value=birth_year,value=birth_year+30,max_value=birth_year+70)
st.write(f"Votre premier investissement a été effecutué à l'Âge de : **{-birth_year+start_year} ans**")

last_year = st.slider("Horizon max Année investissement", min_value=start_year,value=start_year+30,max_value=birth_year+100)

st.write(f"Votre patrimoine sera projeté jusqu'à l'Âge de : **{last_year-birth_year} ans**")
# ------------------ IMMOBILIER ------------------
st.header("🏡 Investissements Immobiliers")
nb_immo = st.number_input("Nombre de biens immobiliers", min_value=0, max_value=100, value=1)
immos = []
for i in range(nb_immo):
    with st.expander(f"Bien immobilier #{i + 1}"):
        montant = st.number_input(f"Montant du bien #{i+1} (€)", value=250000)
        apport = st.number_input(f"Apport personnel (frais inclus) #{i+1} (€)", value=montant*0.1)
        taux = st.number_input(f"Taux de crédit (%) #{i+1}", value=2.0)
        duree = st.number_input(f"Durée crédit (ans) #{i+1}", value=25)
        annee = st.number_input(f"Année d'investissement #{i+1}", value=2022)
        ancien_neuf = st.selectbox(f"Ancien ou Neuf #{i + 1}", ["Ancien", "Neuf"], key=f"ancien_neuf_{i}")
        immos.append({"montant": montant, "apport": apport, "taux": taux / 100, "duree": duree, "annee": annee})


# ------------------ SCPI ------------------
st.header("🏢 SCPI")
type_scpi = st.selectbox("Mode d'investissement", ["Cash","Crédit", "DCA"])
scpi_rendement = st.slider("Rendement annuel SCPI (%)", 0.0, 10.0, 4.5)
scpi_frais = st.slider("Frais d'entrée (%)", 0.0, 15.0, 10.0)
if type_scpi == "Crédit":
    scpi_montant = st.number_input("Montant investi (€)", value=0)
    scpi_annee = st.number_input("Année de souscription", value=2023)
    scpi_duree = st.number_input("Durée crédit SCPI (ans)", value=20)
    scpi_taux = st.number_input("Taux de crédit SCPI (%)", value=2.0)
if type_scpi == "Cash":
    scpi_montant_cash = st.number_input("Montant investi (€)", value=0)
    scpi_annee1 = st.number_input("Année de souscription", value=2023)
else:
    scpi_dca = st.number_input("Versement mensuel SCPI (€)", value=0)
    scpi_annee_dca = st.number_input("Année de démarrage DCA", value=2023)

# ------------------ ETF ------------------
st.header("📊 Investissements Boursiers (ETF)")
etf_classes = ["MSCI World", "S&P500","Nasdaq", "Stoxx 600", "Emerging Markets", "Or", "Obligations", "Private Equity"]

returns = get_annual_returns()
etf_data = {}
for etf in etf_classes:
    with st.expander(f"{etf}"):
        init = st.number_input(f"Apport initial {etf} (€)", value=0, key=etf+"_init")
        dca = st.number_input(f"DCA mensuel {etf} (€)", value=100, key=etf+"_dca")
        rendement = st.slider(f"Rendement annuel attendu {etf} (%)", -20.0, 20.0, returns[etf].iloc[0] , key=etf+"_rendement")
        debut = st.number_input(f"Année de début {etf}", value=2024, key=etf+"_debut")
        etf_data[etf] = {"init": init, "dca": dca, "rendement": rendement / 100, "annee_debut": debut}

# ------------------ CRYPTO ------------------
st.header("🪙 Investissements Crypto")
crypto_assets = ["Bitcoin", "Ethereum", "Altcoins"]
crypto_data = {}
for crypto in crypto_assets:
    with st.expander(f"{crypto}"):
        init = st.number_input(f"Apport initial {crypto} (€)", value=1000, key=crypto+"_init")
        dca = st.number_input(f"DCA mensuel {crypto} (€)", value=10, key=crypto+"_dca")
        rendement = st.slider(f"Rendement annuel attendu {crypto} (%)", -80.0, 100.0, returns[crypto].iloc[0]/4, key=crypto+"_rendement")
        debut = st.number_input(f"Année de début {crypto}", value=2024, key=crypto+"_debut")
        crypto_data[crypto] = {"init": init, "dca": dca, "rendement": rendement / 100, "annee_debut": debut}

# ------------------ PARTICIPATION ------------------
st.header("💼 Participation & Intéressement")
versement_annuel = st.number_input("Montant annuel moyen reçu (€)", value=1000)
annee_debut_part = st.number_input("Année de début de versement", value=2022)
rendement_part = st.slider("Rendement annuel estimé (%)", 0.0, 15.0, 6.0)




projection_years=list(range(start_year,last_year+1))
df = pd.DataFrame(index=projection_years)
df["Age"] = [year - birth_year for year in projection_years]
df["Immobilier"] = 0
df["SCPI"] = 0
df["Bourse"] = 0
df["Crypto"] = 0
df["Participation"] = 0

# Initialisation des colonnes
df["ImmobilierBrut"] = 0
df["Immobilier"] = 0  # Valeur nette = valeur - dette

for bien in immos:
    montant = bien["montant"]
    apport = bien["apport"]
    taux = bien["taux"]
    duree = bien["duree"]
    annee = bien["annee"]
    ancien_neuf = bien.get("ancien_neuf", "Ancien")  # par défaut

    # Frais selon ancien/neuf
    frais = montant * (0.10 if ancien_neuf == "Ancien" else 0.03)
    montant_total = montant + frais
    montant_emprunte = montant_total - apport

    # Utilise la fonction d’amortissement
    df_amort = generer_tableau_amortissement(
        montant_emprunte=montant_emprunte,
        taux_annuel=taux,
        duree_annees=duree,
        annee_debut=annee,
        appreciation=0.02
    )

    # Ajout des valeurs dans le DataFrame principal (si année dans l'intervalle de projection)
    for year, row in df_amort.iterrows():
        if year in df.index:
            df.loc[year, "ImmobilierBrut"] += row["Valeur estimée bien"]
            df.loc[year, "Immobilier"] += row["PatrimoineNet"]

# Calcul du patrimoine net de dette global
if "PatrimoineNet" not in df.columns:
    df["PatrimoineNet"] = 0

# Si d'autres actifs sont pris en compte (ex: Bourse, SCPI, Crypto), additionne-les ici
# Exemple:
# df["PatrimoineNet"] = df["Immobilier"] + df["Bourse"] + df["SCPI"] + df["Crypto"]
# Pour l'instant, on ne considère que l'immobilier net de dette

df["PatrimoineNet"] += df["Immobilier"]



if type_scpi == "Crédit":
    for i in range(int(scpi_duree)):
        year = scpi_annee + i
        if year in df.index:
            croissance = (1 + scpi_rendement / 100) ** (year - scpi_annee)
            net = scpi_montant * croissance * (1 - scpi_frais / 100)
            df.loc[year, "SCPI"] += net

elif type_scpi=="Cash":
    for i in range(int(last_year-start_year+1)):
        year = scpi_annee1 + i
        if year in df.index:
            croissance = (1 + scpi_rendement / 100) ** (year - scpi_annee1)
            net = scpi_montant_cash * croissance * (1 - scpi_frais / 100)
            df.loc[year, "SCPI"] += net

else:
    for year in df.index:
        if year >= scpi_annee_dca:
            mois = (year - scpi_annee_dca + 1) * 12
            total = scpi_dca * mois
            croissance = (1 + scpi_rendement / 100) ** (year - scpi_annee_dca)
            df.loc[year, "SCPI"] += total * croissance * (1 - scpi_frais / 100)


for etf, data in etf_data.items():
    for year in df.index:
        if year >= data["annee_debut"]:
            n_months = (year - data["annee_debut"] + 1) * 12
            montant_dca = data["dca"] * n_months
            croissance = (1 + data["rendement"]) ** (year - data["annee_debut"])

            if year == data["annee_debut"]:
                total = (data["init"] + montant_dca) * croissance
            else:
                total = montant_dca * croissance  # pas d'init ici

            df.loc[year, "Bourse"] += total

for crypto, data in crypto_data.items():
    for year in df.index:
        if year >= data["annee_debut"]:
            n_months = (year - data["annee_debut"] + 1) * 12
            montant_dca = data["dca"] * n_months
            croissance = (1 + data["rendement"]) ** (year - data["annee_debut"])

            # Inclure le montant initial UNE SEULE FOIS à l'année de départ
            if year == data["annee_debut"]:
                total = (data["init"] + montant_dca) * croissance
            else:
                total = montant_dca * croissance  # sans init

            df.loc[year, "Crypto"] += total

for year in df.index:
    if year >= annee_debut_part:
        n = year - annee_debut_part + 1
        valeur = 0
        for i in range(n):
            valeur += versement_annuel * ((1 + rendement_part / 100) ** i)
        df.loc[year, "Participation"] = valeur

# Total
df['Immobilier']=df['PatrimoineNet']
df["Total"] = df[["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]].sum(axis=1)

# ------------------ RÉCAP ------------------
st.header("📋 Récapitulatif du Profil")
st.markdown(f"- 👤 Âge actuel : **{current_age} ans**")
st.markdown(f"- 🏡 Biens immobiliers : **{len(immos)}**")
st.markdown(f"- 🏢 SCPI : **{type_scpi}**")
if type_scpi == "Crédit":
    st.markdown(f"  - Montant : **{scpi_montant} €**, Taux : **{scpi_taux}%**, Durée : **{scpi_duree} ans**")
elif type_scpi == "Cash":
    st.markdown(f"  - Montant investi en Cash: **{scpi_montant_cash} €**, **")

else:
    st.markdown(f"  - DCA mensuel : **{scpi_dca} €**, depuis **{scpi_annee_dca}**")
total_etf = sum([v["init"] + v["dca"] * 12 for v in etf_data.values()])
st.markdown(f"📊 ETF - Total estimé annuel : **{int(total_etf)} €**")
total_crypto = sum([v["init"] + v["dca"] * 12 for v in crypto_data.values()])
st.markdown(f"🪙 Crypto - Total estimé annuel : **{int(total_crypto)} €**")
st.markdown(f"💼 Participation - Versement annuel : **{versement_annuel} €**, rendement : **{rendement_part}%**")

st.header('Global View')
df_view=df.copy()
df_view.rename(columns={"Bourse":"ETFs/Bourses","Participation":"PEE/PERCO"},inplace=True)
st.dataframe(df_view[["Immobilier", "SCPI", "ETFs/Bourses",  "PEE/PERCO","Crypto","Total"]])
# ------------------ RISQUE ------------------
volat_dict = {
    "Immobilier": 0.05,
    "SCPI": 0.07,
    "Bourse": 0.15,
    "Crypto": 0.75,
    "Participation": 0.15
}
df["Volatilite"] = 0.0
df["VaR 95%"] = 0.0
for year in df.index:
    poids = {}
    for asset in volat_dict:
        if df.loc[year, "Total"] > 0:
            poids[asset] = df.loc[year, asset] / df.loc[year, "Total"]
        else:
            poids[asset] = 0
    var_globale = sum((poids[a] * volat_dict[a])**2 for a in volat_dict)
    sigma = sqrt(var_globale)
    df.loc[year, "Volatilite"] = sigma
    df.loc[year, "VaR 95%"] = -norm.ppf(0.05) * sigma * df.loc[year, "Total"]

# ------------------ GRAPHIQUES ------------------
st.header("📊 Projections et Risque")
#df=df[["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]]

st.line_chart(df["Total"], height=250)
st.area_chart(df[["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]], height=250)
max_age=last_year-birth_year-1 #,int(df["Age"].max())) int(df["Age"].max()) #
st.write(max_age)
age_select = st.slider("🔢 Voir la répartition à l'âge de :", int(df["Age"].min()), max_value=max_age, value=current_age)

# Message sur le patrimoine net total à cet âge
patrimoine_total = df.loc[birth_year + age_select, "Total"]
st.success(f"💰 À {age_select} ans, le patrimoine net estimé est de **{patrimoine_total:,.0f} €**.")

if (birth_year + age_select) in df.index:
    pie_df = df.loc[birth_year + age_select, ["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]]
    fig = px.pie(values=pie_df.values, names=pie_df.index, title=f"Répartition à {age_select} ans")
    st.plotly_chart(fig)


# ------------------ LIBERTÉ FINANCIÈRE ------------------
st.header("💸 Liberté financière")
target_income =st.slider("Besoin annuel pour vivre libre (€)", 12000,100000,36000)
df["Revenus 4%"] = df["Total"] * 0.04
freedom = df[df["Revenus 4%"] >= target_income]
if not freedom.empty:
    age_libre = int(freedom["Age"].iloc[0])
    annee_libre = int(freedom.index[0])
    st.success(f"🌟 Liberté financière atteinte à **{age_libre} ans** (en {annee_libre})")
else:
    st.warning("❌ Liberté financière non atteinte dans la période simulée.")
st.line_chart(df[["Revenus 4%"]])

# ------------------ RECOMMANDATIONS ------------------
st.header("🧐 Recommandations Dynamiques")
alloc = df.iloc[-1][["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]]
alloc_pct = (alloc / alloc.sum() * 100).round(1)
st.write("🌐 Allocation finale du portefeuille :")
st.write(alloc_pct.astype(str) + " %")
warnings = []
if alloc_pct["Crypto"] > 30:
    warnings.append("🚨 Trop de crypto (>30%) : volatilité très élevée.")
if alloc_pct["Immobilier"] < 10:
    warnings.append("🏠 Faible part d'immobilier : manque de stabilité.")
if alloc_pct["Bourse"] < 20:
    warnings.append("📉 Exposition boursière faible : croissance long terme sous-exploitée.")
if alloc_pct["SCPI"] > 50:
    warnings.append("🏢 SCPI > 50% : attention à la fiscalité et illiquidité.")
if not warnings:
    st.success("✅ Portefeuille bien diversifié.")
else:
    for w in warnings:
        st.warning(w)

# ------------------ RISQUE VISUEL ------------------

import streamlit as st
import plotly.graph_objects as go
st.subheader("📈 Risque du portefeuille (VaR 95% et Volatilité)")

# Création du graphique avec deux axes y
fig = go.Figure()

# Trace VaR 95% sur l'axe y principal
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["VaR 95%"],
    name="VaR 95%",
    yaxis="y1",
    line=dict(color="red")
))

# Trace Volatilité sur l'axe y secondaire
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Volatilite"],
    name="Volatilité",
    yaxis="y2",
    line=dict(color="blue", dash="dot")
))

# Configuration des axes
fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(
        title="VaR 95%",
        #titlefont=dict(color="red"),
        tickfont=dict(color="red")
    ),
    yaxis2=dict(
        title="Volatilité",
        #titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
        overlaying="y",
        side="right"
    ),
    legend=dict(x=0.01, y=0.99),
    height=500
)

# Affichage dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# ------------------ EXPORTS PDF / EXCEL ------------------
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# import pandas as pd
# from fpdf import FPDF

st.header("📤 Export du rapport et des données")
#
# # Génération Excel
# excel_buffer = BytesIO()
# df.to_excel(excel_buffer, index=True, engine='openpyxl')
# excel_data = excel_buffer.getvalue()
#
# st.download_button("📊 Télécharger les données Excel", data=excel_data,
#                    file_name="patrimoine_simulation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
#
# # Graphique pour le PDF
# fig, ax = plt.subplots()
# df[["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]].plot.area(ax=ax)
# plt.title("Évolution du patrimoine par classe d’actif")
# plt.xlabel("Année")
# plt.ylabel("Montant (€)")
# fig_buffer = BytesIO()
# plt.savefig(fig_buffer, format="png")
# fig_buffer.seek(0)
#
#
#
# pdf = PDF()
# pdf.add_page()
#
# pdf.section_title("📋 Profil Utilisateur")
# pdf.section_body(f"Année de naissance : {birth_year}\nÂge actuel : {current_age} ans\n")
#
# pdf.section_title("💼 Participation & Intéressement")
# pdf.section_body(f"Montant annuel : {versement_annuel} €, depuis {annee_debut_part}, rendement : {rendement_part}%")
#
# pdf.section_title("📊 Répartition finale")
# repartition_finale = df.iloc[-1][["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]]
# txt = "\n".join([f"{k} : {int(v):,} €" for k, v in repartition_finale.items()])
# pdf.section_body(txt)
# freedom_age=age_libre
# pdf.section_title("🔁 Liberté Financière")
# if freedom_age:
#     pdf.section_body(f"Atteinte à {freedom_age} ans (année {freedom_age + birth_year})")
# else:
#     pdf.section_body("Pas atteinte dans la période simulée.")
#
# # Ajout du graphique
# pdf.image(fig_buffer, x=10, y=None, w=180)
#
# # Export du PDF
# pdf_buffer = BytesIO()
# pdf.output(pdf_buffer)
# pdf_buffer.seek(0)
#
# st.download_button("📄 Télécharger le rapport PDF", data=pdf_buffer, file_name="rapport_patrimoine.pdf", mime="application/pdf")
