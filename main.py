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
import requests
import datetime as dt
import sqlite3
from datetime import datetime, date
# CONFIGURATION
st.set_page_config(page_title="Simulateur Patrimoine", layout="wide")
st.markdown(f"- 👤 Mis à disposition par Michael V. **")
st.markdown("""
> ⚠️ **Disclaimer**
> 
> This tool does not constitute financial advice or a recommendation to take financial risks.  
> Always do your own research before making any investment decisions.
""")
# === 1. Fonction pour obtenir IP et localisation
def get_ip_and_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        ip = res.get("ip", "N/A")
        city = res.get("city", "N/A")
        country = res.get("country", "N/A")
        return ip, city, country
    except:
        return "N/A", "N/A", "N/A"

# === 2. Connexion à la base SQLite
conn = sqlite3.connect("visiteurs.db", check_same_thread=False)
c = conn.cursor()

# === 3. Créer la table si elle n'existe pas
c.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ip TEXT,
        city TEXT,
        country TEXT
    )
""")

# === 4. Ajouter une nouvelle visite
ip, city, country = get_ip_and_location()
timestamp = datetime.now().isoformat()
c.execute("INSERT INTO visits (timestamp, ip, city, country) VALUES (?, ?, ?, ?)",
          (timestamp, ip, city, country))
conn.commit()

# === 5. Stats globales et journalières
c.execute("SELECT COUNT(*) FROM visits")
visit_count = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM visits WHERE DATE(timestamp) = ?", (date.today().isoformat(),))
daily_visits = c.fetchone()[0]

# === 6. Affichage dans la sidebar
st.sidebar.markdown("## 📊 Statistiques de visites")
st.sidebar.markdown(f"👥 **Visites totales** : `{visit_count}`")
st.sidebar.markdown(f"📅 **Aujourd’hui** : `{daily_visits}`")
st.sidebar.markdown(f"🧭 **Localisation** : `{city}, {country}`")
st.sidebar.markdown(f"🕒 **Dernière visite** : `{timestamp[:19]}`")

# === 7. (Optionnel) Dernières visites
with st.expander("🔍 Dernières visites"):
    rows = conn.execute("SELECT timestamp, city, country FROM visits ORDER BY id DESC LIMIT 5").fetchall()
    st.table(rows)

# === 8. Fermer la connexion
conn.close()

st.title("💰 Simulateur de Patrimoine")

# ------------------ PROFIL ------------------
st.header("👤 Profil Utilisateur")
birth_year = st.number_input("Votre année de naissance", min_value=1925, max_value=datetime.now().year, value=1990)


current_year = datetime.now().year
current_age = current_year - birth_year
st.write(f"Âge actuel : **{current_age} ans**")
start_year = st.number_input(f"Année du premier investissement", min_value=birth_year,value=birth_year+30,max_value=birth_year+70)
st.write(f"Votre premier investissement a été effecutué à l'Âge de : **{-birth_year+start_year} ans**")

last_year = st.slider("A quel Horizon souhaitez-vous projeter votre patrimoine?", min_value=start_year,value=current_year+15,max_value=birth_year+100)

st.write(f"Votre patrimoine sera projeté jusqu'à l'Âge de : **{last_year-birth_year} ans**")
# ------------------ IMMOBILIER ------------------
st.header("🏡 Investissements Immobiliers")
nb_immo = st.number_input("Nombre de biens immobiliers", min_value=0, max_value=20, value=1)
immos = []
for i in range(nb_immo):
    with st.expander(f"Bien immobilier #{i + 1}"):
        montant = st.number_input(f"Montant du bien #{i+1} (€)", value=100000)
        apport = st.number_input(f"Apport personnel (frais inclus) #{i+1} (€)",value=0)
        taux = st.number_input(f"Taux de crédit (%) #{i+1}", value=2.0)
        duree = st.number_input(f"Durée crédit (ans) #{i+1}", value=25)
        annee = st.number_input(f"Année d'investissement #{i+1}", value=current_year)
        ancien_neuf = st.selectbox(f"Ancien ou Neuf #{i + 1}", ["Ancien", "Neuf"], key=f"ancien_neuf_{i}")
        immos.append({"montant": montant, "apport": apport, "taux": taux / 100, "duree": duree, "annee": annee})


# ------------------ SCPI ------------------
st.header("🏢 SCPI")
st.info("https://www.economie.gouv.fr/particuliers/investir-dans-limmobilier/scpi-investissez-dans-limmobilier-avec-un-placement#")
type_scpi = st.selectbox("Mode d'investissement", ["Cash", "DCA"])
scpi_rendement = st.slider("Rendement annuel SCPI (%)", 0.0, 10.0, 4.5)
scpi_frais = st.slider("Frais d'entrée (%)", 0.0, 15.0, 10.0)
if type_scpi == "Crédit":
    scpi_montant = st.number_input("Montant investi (€)", value=0)
    scpi_annee = st.number_input("Année de souscription", value=current_year)
    scpi_duree = st.number_input("Durée crédit SCPI (ans)", value=0)
    scpi_taux = st.number_input("Taux de crédit SCPI (%)", value=3.0)
if type_scpi == "Cash":
    scpi_montant_cash = st.number_input("Montant investi (€)", value=0)
    scpi_annee1 = st.number_input("Année de souscription", value=current_year)
else:
    scpi_dca = st.number_input("Versement mensuel SCPI (€)", value=0)
    scpi_annee_dca = st.number_input("Année de démarrage DCA", value=current_year)

# ------------------ ETF ------------------
st.header("📊 Investissements Boursiers (ETF)")
st.markdown("rendements par défaut, source yahoo finance (médiane annuelle historique)**")
etf_classes = ["MSCI World", "S&P500","Nasdaq", "Stoxx 600", "Emerging Markets", "Or", "Obligations", "Private Equity"]
etf_classes=list(etf_index().keys())[:-3]

returns = get_annual_returns()
etf_data = {}
for etf in etf_classes:
    with st.expander(f"{etf}"):
        init = st.number_input(f"Apport initial {etf} (€)", value=0, key=etf+"_init")
        dca = st.number_input(f"DCA mensuel {etf} (€)", value=0, key=etf+"_dca")
        rendement = st.slider(f"Rendement annuel attendu {etf} (%)", -20.0, 20.0, returns[etf] , key=etf+"_rendement")
        debut = st.number_input(f"Année de début {etf}", value=2024, key=etf+"_debut")
        etf_data[etf] = {"init": init, "dca": dca, "rendement": rendement / 100, "annee_debut": debut}

# ------------------ CRYPTO ------------------
st.header("🪙 Investissements Crypto")
crypto_assets = ["Bitcoin", "Ethereum", "Altcoins"]
crypto_data = {}
for crypto in crypto_assets:
    with st.expander(f"{crypto}"):
        init = st.number_input(f"Apport initial {crypto} (€)", value=0, key=crypto+"_init")
        dca = st.number_input(f"DCA mensuel {crypto} (€)", value=0, key=crypto+"_dca")
        rendement = st.slider(f"Rendement annuel attendu {crypto} (%)", -80.0, 100.0, returns[crypto]/4, key=crypto+"_rendement")
        debut = st.number_input(f"Année de début {crypto}", value=current_year, key=crypto+"_debut")
        crypto_data[crypto] = {"init": init, "dca": dca, "rendement": rendement / 100, "annee_debut": debut}

# ------------------ PARTICIPATION ------------------
st.header("💼 Participation & Intéressement")
versement_annuel = st.number_input("Montant annuel moyen reçu (€)", value=0)
annee_debut_part = st.number_input("Année de début de versement", value=current_year)
rendement_part = st.slider("Rendement annuel estimé (%)", 0.0, 15.0, 7.0)

# ------------------ PASSIF- VOITURES ------------------
st.header("💼 Voitures- Montres...")
valeur_passif = st.number_input("Valorisation (€)", value=0)
annee_debut_passif = st.number_input("Année d'achat", value=current_year)
rendement_passif = st.slider("Rendement annuel passif estimé (%)", -20.0, 10.0, -8.0)

# ------------------ EPARGNE DE SECURITE ------------------
st.header("💼 EPARGNE DE SECURITE")
valeur_epargne_securite = st.number_input("Epargne de sécurité (€)", value=0)
annee_debut_epargne_securite = st.number_input("Année ", value=current_year)
rendement_epargne_securite = st.slider("Rendement annuel Livret (%)", 0.0, 7.0, 2.0)

# ------------------ CREDIT CONSOMMATIONS ------------------
st.header("💼 CREDITS CONSOMMATIONS")
montant_credit_conso = st.number_input(f"Montant du credit conso (€)", value=0)
taux_credit_conso = st.number_input(f"Taux de crédit conso (%)", value=4.0)
duree_credit_conso = st.number_input(f"Durée crédit conso (ans)", value=5)
annee_credit_conso = st.number_input(f"Année du pret conso", value=current_year)

projection_years=list(range(start_year,last_year+1))
df = pd.DataFrame(index=projection_years)
df["Age"] = [year - birth_year for year in projection_years]
df["Immobilier"] = 0
df["SCPI"] = 0
df["Bourse"] = 0
df["Crypto"] = 0
df["Participation"] = 0
df["Others"] = 0
df['Livrets']=0
df['Conso']=0

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

#Livrets
for i in range(50):
    year = annee_debut_epargne_securite + i
    if year in df.index:
        croissance = (1 + scpi_rendement / 100) ** (year - annee_debut_epargne_securite)
        net = valeur_epargne_securite * croissance * (1 - rendement_epargne_securite / 100)
        df.loc[year, "Livrets"] += net

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
        total = 0.0
        if year >= scpi_annee_dca:
            for past_year in range(scpi_annee_dca, year + 1):
                for month in range(12):
                    # Nombre de mois depuis ce versement jusqu'à l'année courante
                    months_since = (year - past_year) * 12 + (11 - month)  # 11 au lieu de 12 pour 0-based index
                    years_since = months_since / 12

                    montant_net = scpi_dca * (1 - scpi_frais / 100)
                    total += montant_net * ((1 + scpi_rendement / 100) ** years_since)

            df.loc[year, "SCPI"] += total


#sauvegarde poids etfs
# 1. Calcul de la valeur totale investie par ETF (initial + DCA estimé sur 1 an par exemple)
poids_etfs = {}
total_investi = 0

for etf, data in etf_data.items():
    # Tu peux affiner ce calcul avec la durée réelle d'investissement, ici on suppose 1 an
    total_etf = data["init"] + data["dca"] * 12
    poids_etfs[etf] = total_etf
    total_investi += total_etf

# 2. Conversion en pourcentages
for etf in poids_etfs:
    poids_etfs[etf] = round(poids_etfs[etf] / total_investi, 4) if total_investi > 0 else 0.0




#compute etfs investments amount
for _, data in etf_data.items():
    for year in df.index:
        if year >= data["annee_debut"]:
            total = 0.0

            # 1. Montant initial investi à l'année de départ
            years_since_start = year - data["annee_debut"]
            total += data["init"] * ((1 + data["rendement"]) ** years_since_start)

            # 2. DCA mensuel : chaque mois est capitalisé individuellement
            for past_year in range(data["annee_debut"], year + 1):
                months = 12
                for month in range(months):
                    months_since = (year - past_year) * 12 + (12 - month)
                    years_since = months_since / 12
                    total += data["dca"] * ((1 + data["rendement"]) ** years_since)

            df.loc[year, "Bourse"] += total


#compute crypto investments amount
for _, data in crypto_data.items():
    for year in df.index:
        if year >= data["annee_debut"]:
            total = 0.0

            # 1. Montant initial investi à l'année de départ
            years_since_start = year - data["annee_debut"]
            total += data["init"] * ((1 + data["rendement"]) ** years_since_start)

            # 2. DCA mensuel : chaque mois est capitalisé individuellement
            for past_year in range(data["annee_debut"], year + 1):
                months = 12
                for month in range(months):
                    months_since = (year - past_year) * 12 + (12 - month)
                    years_since = months_since / 12
                    total += data["dca"] * ((1 + data["rendement"]) ** years_since)

            df.loc[year, "Crypto"] += total

#compute participation/interessement
for year in df.index:
    if year >= annee_debut_part:
        n = year - annee_debut_part + 1
        valeur = 0
        for i in range(n):
            valeur += versement_annuel * ((1 + rendement_part / 100) ** i)
        df.loc[year, "Participation"] = valeur

# Total
df['Immobilier']=df['PatrimoineNet']

#Others
for i in range(40):
    year = annee_debut_passif + i
    if year in df.index:
        croissance = (1 + rendement_passif / 100) ** (year - annee_debut_passif)
        net = valeur_passif * croissance
        if net <0:
            net=0

        df.loc[year, "Others"] += net

df["Total"] = df[["Livrets","Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"]].sum(axis=1)
df=df[df['Total']!=0]
# ------------------ RÉCAP ------------------
st.header("📋 Récapitulatif du Profil")
st.markdown(f"- 👤 Âge actuel : **{current_age} ans**")
st.markdown(f"- Epargne de sécurité :**{valeur_epargne_securite} €**")
# 💰 Somme des pret immos
somme_pret = sum([bien["montant"] for bien in immos])
st.markdown(f"- 🏡 Biens immobiliers : **{len(immos)}** & ### 💰 avec un pret total estimé de : **{somme_pret:,.0f} €**")
st.markdown(f"- 🏢 SCPI : **{type_scpi}**")
if type_scpi == "Crédit":
    st.markdown(f"  -- Montant : **{scpi_montant} €**, Taux : **{scpi_taux}%**, Durée : **{scpi_duree} ans**")
elif type_scpi == "Cash":
    st.markdown(f"  - Montant investi en Cash: **{scpi_montant_cash} €**, **")

else:
    st.markdown(f"  - DCA mensuel : **{scpi_dca} €**, depuis **{scpi_annee_dca}**")
total_etf = sum([v["dca"] * 12 for v in etf_data.values()])
total_etf_apport_inital = sum([v["init"]  for v in etf_data.values()])

st.markdown(f"📊 ETF - Total estimé annuel : **{int(total_etf)} €/an avec un apport initial de {total_etf_apport_inital} €**")
st.markdown(f"📊 **Poids de chaque ETF dans le portefeuille : {poids_etfs}**")

total_crypto = sum([ v["dca"] * 12 for v in crypto_data.values()])
init_crypto = sum([ v["init"]  for v in crypto_data.values()])
st.markdown(f"🪙 Crypto - Total estimé annuel : **{int(total_crypto)} €/an avec un apport initial de {init_crypto} € **")
st.markdown(f"💼 Participation - Versement annuel : **{versement_annuel} €**, rendement : **{rendement_part}%**")

st.header('Global View')
df_view=df.copy()
df_view.rename(columns={"Bourse":"ETFs/Bourses","Participation":"PEE/PERCO"},inplace=True)
# Arrondir à 0 chiffre après la virgule et ajouter les séparateurs de milliers
df_formatted=df_view[["Livrets","Immobilier", "SCPI", "ETFs/Bourses",  "PEE/PERCO","Crypto","Others","Total"]]

df_formatted = df_formatted.replace([np.inf, -np.inf], np.nan).dropna()
df_formatted = df_formatted.round(0).astype(int).applymap(lambda x: f"{x:,}".replace(",", " "))


#print(df_formatted)
st.dataframe(df_formatted)
# ------------------ RISQUE ------------------
volat_dict = {
    "Immobilier": 0.06,
    "SCPI": 0.1,
    "Bourse": 0.17,
    "Crypto": 0.80,
    "Participation": 0.17,
    "Others":0.10,
     "Livrets":0
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
    df.loc[year, "Volatilite"] = sigma*100
    df.loc[year, "VaR 95%"] = -norm.ppf(0.05) * sigma * df.loc[year, "Total"]

# ------------------ GRAPHIQUES ------------------
st.header("📊 Projections et Risque")
#df=df[["Immobilier", "SCPI", "Bourse", "Crypto", "Participation"]]

st.line_chart(df["Total"], height=250)
st.area_chart(df[["Livrets","Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"]], height=250)
max_age=last_year-birth_year-1 #,int(df["Age"].max())) int(df["Age"].max()) #

age_select = st.slider("🔢 Voir la répartition à l'âge de :", int(df["Age"].min()), max_value=max_age, value=current_age)

# Message sur le patrimoine net total à cet âge
patrimoine_total = df.loc[birth_year + age_select, "Total"]
st.success(f"💰 À {age_select} ans, le patrimoine net estimé est de **{patrimoine_total:,.0f} €**.")

# if (birth_year + age_select) in df.index:
#     pie_df = df.loc[birth_year + age_select, ["Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"]]
#     fig = px.pie(values=pie_df.values, names=pie_df.index, title=f"Répartition à {age_select} ans")
#     st.plotly_chart(fig)
#
#     st.plotly_chart(px.bar(pie_df,x=pie_df.index,y=["Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"], title=f"Répartition à {age_select} ans)
#     st.plotly_chart(fig)

if (birth_year + age_select) in df.index:
    pie_df = df.loc[birth_year + age_select, ["Livrets","Immobilier", "SCPI", "Bourse", "Crypto", "Participation", "Others"]]

    # Deux colonnes pour affichage côte à côte
    col1, col2 = st.columns(2)

    with col2:
        fig_pie = px.pie(
            values=pie_df.values,
            names=pie_df.index,
            title=f"Répartition à {age_select} ans"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col1:
        fig_bar = px.bar(
            x=pie_df.index,
            y=pie_df.values,
            #text=pie_df.values.round(0),  # Ajout des valeurs comme texte

            #text="outside",  # Position au-dessus des barres
            labels={"x": "Type d'investissement", "y": "Montant"},
            title=f"Répartition à {age_select} ans"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    #commentaires
    st.header(f"🧐 Recommandations Allocation projété du portefeuille : à {age_select} ans 🧐")
    alloc =pie_df # df.iloc[-1][["Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"]]
    alloc_pct = (alloc / alloc.sum() * 100).round(1)
    #st.write(f"🌐 Allocation projété du portefeuille : à {age_select} ans")
    st.write(alloc.round(0).T)
    #st.write(alloc_pct.astype(str) + " %", alloc)
    warnings = []
    if alloc_pct["Crypto"] > 30:
        warnings.append("🚨 Trop de crypto (>30%) : volatilité très élevée.")
    if alloc_pct["Immobilier"] < 10:
        warnings.append("🏠 Faible part d'immobilier : manque de stabilité.")
    if alloc_pct["Bourse"] < 20:
        warnings.append("📉 Exposition boursière faible : croissance long terme sous-exploitée.")
    if alloc_pct["SCPI"] > 50:
        warnings.append("🏢 SCPI > 50% : attention à la fiscalité et illiquidité.")
    if alloc_pct["Others"] > 50:
        warnings.append("🚨 Others > 50% : attention lié a la illiquidité et de forte dépréciation 🚨.")
    if alloc_pct["Participation"] > 50:
        warnings.append("🏢 Participation > 50% : risque de concentration auprès d'un seul assureur, transfert sur les fonds disponibles depuis le PEE?")
    if not warnings:
        st.success("✅ Portefeuille bien diversifié.")
    else:
        for w in warnings:
            st.warning(w)

# ------------------ LIBERTÉ FINANCIÈRE ------------------
st.header("💸 Liberté financière non ajusté à l'inflation")
target_income =st.slider("Besoin annuel pour vivre libre (€)", 12000,100000,36000)
df["Revenus 4%"] = df["Total"] * 0.045
freedom = df[df["Revenus 4%"] >= target_income]
if not freedom.empty:
    age_libre = int(freedom["Age"].iloc[0])
    annee_libre = int(freedom.index[0])
    st.success(f"🌟 Liberté financière atteinte à **{age_libre} ans** (en {annee_libre})")
else:
    st.warning("❌ Liberté financière non atteinte dans la période simulée.")
#st.bar_chart(df[["Revenus 4%"]]/12,title="Estmation de revenus mensuelle liberté financière")
import plotly.graph_objects as go

revenus_mensuels = (df["Revenus 4%"] / 12).round(0)
import plotly.graph_objects as go
import numpy as np

# Revenus mensuels arrondis
revenus_mensuels = (df["Revenus 4%"] / 12).round(0)

# Calcul des âges (x-axis)
ages = df.index - birth_year

# Paramètres
inflation_rate = 0.025
seuil_initial = target_income / 12

# Calcul du seuil ajusté à l'inflation pour chaque âge
seuils_inflation = [seuil_initial * ((1 + inflation_rate) ** i) for i in range(len(ages))]

fig = go.Figure()

# Barres des revenus
fig.add_trace(go.Bar(
    x=ages,
    y=revenus_mensuels,
    name="Revenus mensuels (4%)",
    marker_color='mediumseagreen',
    text=revenus_mensuels.astype(str) + " €",
    textposition='outside',
    insidetextanchor="start"
))

# Ligne rouge du seuil de liberté ajusté à l'inflation
fig.add_trace(go.Scatter(
    x=ages,
    y=seuils_inflation,
    mode="lines",
    name="Seuil ajusté (inflation 2.5%)",
    line=dict(color="red", dash="dash")
))

# Annotation à la fin de la ligne
fig.add_annotation(
    x=ages[-1],
    y=seuils_inflation[-1],
    text=f"Seuil ajusté : {int(seuils_inflation[-1])} €",
    showarrow=False,
    font=dict(color="red", size=12),
    bgcolor="white"
)

# Layout
fig.update_layout(
    title="Estimation des revenus mensuels (avec seuil ajusté à l'inflation)",
    xaxis_title="Âge",
    yaxis_title="Montant mensuel (€)",
    uniformtext_minsize=8,
    uniformtext_mode='show',
    bargap=0.2
)

st.plotly_chart(fig, use_container_width=True)


# import plotly.graph_objects as go
# 
# fig = go.Figure()
# df["Liberté Financière Mensuelle"]=df["Revenus 4%"]/12
# # Courbe des revenus à 4%
# fig.add_trace(go.Scatter(
#     x=df.index,
#     y=df["Revenus 4%"],
#     name="Revenus 4%",
#     mode="lines",
#     line=dict(color="royalblue")
# ))
# 
# # Exemple : ajout du montant mensuel de liberté financière (doit exister dans le DataFrame)
# fig.add_trace(go.Scatter(
#     x=df.index,
#     y=df["Liberté Financière Mensuelle"],
#     name="Liberté financière mensuelle",
#     mode="lines",
#     line=dict(color="darkorange"),
#     yaxis="y2"
# ))
# 
# # Mise en forme des deux axes Y
# fig.update_layout(
#     title="Projection dans le temps",
#     xaxis_title="Âge ou Année",
#     yaxis=dict(
#         title="Revenus 4%",
#         titlefont=dict(color="royalblue"),
#         tickfont=dict(color="royalblue")
#     ),
#     yaxis2=dict(
#         title="Liberté Financière Mensuelle",
#         titlefont=dict(color="darkorange"),
#         tickfont=dict(color="darkorange"),
#         overlaying="y",
#         side="right"
#     ),
#     legend=dict(x=0.01, y=0.99)
# )
# 
# st.plotly_chart(fig, use_container_width=True)


# ------------------ RECOMMANDATIONS ------------------
st.header("🧐 Recommandations Dynamiques")
alloc = df.iloc[-1][["Immobilier", "SCPI", "Bourse", "Crypto", "Participation","Others"]]
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
        title="Volatilité (%)",
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

# Génération Excel
excel_buffer = BytesIO()
df.to_excel(excel_buffer, index=True, engine='openpyxl')
excel_data = excel_buffer.getvalue()

st.download_button("📊 Télécharger les données Excel", data=excel_data,
                   file_name="patrimoine_simulation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("### 💖 Soutenir l'app")
st.sidebar.markdown("Vous aimez cette app ?")
st.sidebar.markdown("""
<a href="https://www.buymeacoffee.com/schadmichael" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50">
</a>
""", unsafe_allow_html=True)
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


#rajout

# st.header("📐 Optimisation de portefeuille (Frontière efficiente)")
#
# # === Sélection des classes d’actifs à optimiser ===
# actifs_opt = ["Livrets", "Immobilier", "SCPI", "Bourse", "Crypto", "Participation", "Others"]
# mean_returns = np.array([returns.get(k, 3.0)/100 for k in actifs_opt])  # Exemple : rendement moyen par actif
# vols = np.array([volat_dict[k] for k in actifs_opt])  # Volatilité annuelle estimée
# cov_matrix = np.diag(vols**2)  # Hypothèse simpliste : pas de corrélations (sinon tu peux estimer corrélation réelle)
#
# # === Génération de portefeuilles ===
# n_portfolios = 10_000
# results = np.zeros((3, n_portfolios))
# weights_record = []
#
# for i in range(n_portfolios):
#     weights = np.random.random(len(actifs_opt))
#     weights /= np.sum(weights)
#
#     portfolio_return = np.dot(weights, mean_returns)
#     portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#     sharpe_ratio = portfolio_return / portfolio_vol
#
#     results[0, i] = portfolio_return
#     results[1, i] = portfolio_vol
#     results[2, i] = sharpe_ratio
#     weights_record.append(weights)
#
# # === Résultats sous forme de DataFrame ===
# results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe"])
# for i, name in enumerate(actifs_opt):
#     results_df[name] = [w[i] for w in weights_record]
#
# # === Trouver le portefeuille optimal ===
# max_sharpe_idx = results_df["Sharpe"].idxmax()
# opt_portfolio = results_df.iloc[max_sharpe_idx]
#
# #affichage graphique
# fig_opt = px.scatter(results_df, x="Volatility", y="Return", color="Sharpe",
#                      hover_data=actifs_opt, title="Frontière efficiente - Simulation")
#
# fig_opt.add_trace(go.Scatter(
#     x=[opt_portfolio["Volatility"]],
#     y=[opt_portfolio["Return"]],
#     mode="markers+text",
#     marker=dict(color='red', size=12, symbol="star"),
#     name="Portefeuille optimal",
#     text=["Max Sharpe"],
#     textposition="top center"
# ))
# st.plotly_chart(fig_opt, use_container_width=True)
#
# st.subheader("📌 Allocation optimale (max Sharpe ratio)")
# for a in actifs_opt:
#     st.markdown(f"- **{a}** : {opt_portfolio[a]*100:.1f}%")
# st.markdown(f"📈 **Rendement attendu** : {opt_portfolio['Return']*100:.2f} %")
# st.markdown(f"📉 **Volatilité estimée** : {opt_portfolio['Volatility']*100:.2f} %")
# st.markdown(f"📊 **Ratio de Sharpe** : {opt_portfolio['Sharpe']:.2f}")
st.header("📘 Validation/Optimisation du portefeuille d'ETFs avec/sans contraintes")

# === Paramètres utilisateur ===
etf_tickers = ["SPY", "QQQ", "VEA", "VWO", "BND", "GLD", "VNQ"]
etf_classes_inverse = {v: k for k, v in etf_index().items()}

etf_tickers=list(etf_classes_inverse.keys())[:-3] #etf_index().keys()[:-3]
#st.markdown("Sélectionne la période de reférences pour les rendements historiques :")

# years = st.selectbox("Horizon historique", [5, 10,15,20,25,30,35,40])
# end_date = datetime.today()
# start_date = end_date - dt.timedelta(days=years*365)
#
#
# st.write(f"📅 Analyse sur la période : {start_date.date()} ➜ {end_date.date()}")
from datetime import  date

st.markdown("### 📅 Choisis ta période de référence historique pour les rendements")

# Valeurs par défaut
default_start = date(1930, 1, 1)
default_end = date(2024, 12, 31)

# Sélection manuelle de la plage de dates
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📆 Date de début", value=default_start, min_value=date(1900, 1, 1), max_value=date.today())
with col2:
    end_date = st.date_input("📆 Date de fin", value=default_end, min_value=start_date, max_value=date.today())

if start_date >= end_date:
    st.error("❌ La date de début doit être antérieure à la date de fin.")
else:
    st.success(f"📅 Analyse sur la période : `{start_date.strftime('%d/%m/%Y')}` ➜ `{end_date.strftime('%d/%m/%Y')}`")

# === Contrainte personnalisée sur chaque ETF
st.markdown("🔧 Définis des contraintes de poids pour chaque ETF (en %) :")
constraints_df = pd.DataFrame(index=etf_tickers, columns=["Min", "Max"])
for etf_ in etf_tickers:
    etf=etf_classes_inverse[etf_]
    c1, c2 = st.columns(2)
    with c1:
        min_val = st.slider(f"Min {etf}({etf_})", 0.0, 1.0, 0.0, 0.05, key=f"min_{etf}({etf_})")
    with c2:
        max_val = st.slider(f"Max {etf}({etf_})", 0.0, 1.0, 1.0, 0.05, key=f"max_{etf}({etf_})")
    constraints_df.loc[etf_] = [min_val, max_val]

# === Téléchargement des données
@st.cache_data
def load_etf_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data.dropna()

data = load_etf_data(etf_tickers, start_date, end_date)
log_returns = np.log(data / data.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252

# === Optimisation avec contraintes
from scipy.optimize import minimize

def neg_sharpe(weights, mean_r, cov):
    port_r = np.dot(weights, mean_r)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -port_r / port_vol if port_vol != 0 else np.inf

nb_assets = len(etf_tickers)
bounds = [(constraints_df.loc[etf, "Min"], constraints_df.loc[etf, "Max"]) for etf in etf_tickers]
x0 = np.array([1.0 / nb_assets] * nb_assets)
constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

opt_result = minimize(neg_sharpe, x0,
                      args=(mean_returns.values, cov_matrix.values),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)

# === Résultats de l'optimisation
if opt_result.success:
    opt_weights = opt_result.x
    port_r = np.dot(opt_weights, mean_returns)
    port_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    sharpe = port_r / port_vol

    st.success("✅ Optimisation réussie")

    st.subheader("📌 Allocation optimale (avec contraintes)")
    cols = st.columns(4)  # Crée 3 colonnes
    # === Affichage graphique
    fig_allocation = px.pie(names=etf_tickers, values=opt_weights, title="📊 Répartition optimale du portefeuille ETF")
    st.plotly_chart(fig_allocation, use_container_width=True)

    st.subheader("Résultats clés")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"📈 **Rendement attendu**\n\n`{port_r * 100:.2f} %`")

    with col2:
        st.markdown(f"📉 **Volatilité estimée**\n\n`{port_vol * 100:.2f} %`")

    with col3:
        st.markdown(f"📊 **Ratio de Sharpe**\n\n`{sharpe:.2f}`")


    for i, etf in enumerate(etf_tickers):
        col = cols[i % 4]  # Sélectionne la colonne de manière cyclique
        with col:
            st.markdown(f"- **{etf_classes_inverse[etf]}** : {opt_weights[i] * 100:.1f}%")



    # === Bouton "Utiliser cette allocation"
    st.title("Test & Validation")
    if st.button(f"📦 Cliquer ici pour Utiliser cette Nouvelle allocation pour simuler un portefeuille"): #: {dict(zip(etf_tickers, opt_weights))}
        st.session_state.etf_allocation = dict(zip(etf_tickers, opt_weights))
        st.success(f"✅ Cliquer ici pour Tester et projeter la nouvelle Allocation proposée ci-dessus!")
        title_statment = "📊 Simulation de  l'allocation recalculée vs Évolution réelle du portefeuille"
    elif st.button(f"📦 Tester et projeter votre allocation actuelle composée de : {dict(zip(etf_tickers, poids_etfs.values()))}"):
        title_statment="📊 Simulation de votre allocation actuelle vs Évolution réelle du portefeuille"
        st.session_state.etf_allocation = dict(zip(etf_tickers, poids_etfs.values()))
        st.success("✅ Allocation de départ enregistrée pour simulation !")


else:
    st.error("❌ Optimisation échouée — vérifie les contraintes.")
########################################################################
import matplotlib.pyplot as plt

# === Paramètres de simulation
years = 20
n_simulations = 500
initial_capital = st.number_input("💰 Capital initial (€)", value=10000, step=1000)
nb_month_projection = st.slider (" Sur combien de mois souhaitez projeter votre analyse?", min_value=12,value=12, max_value=30*12)

# On vérifie si une allocation a été enregistrée
if "etf_allocation" in st.session_state:
    # alloc = st.session_state.etf_allocation
    # weights = np.array([alloc[ticker] for ticker in etf_tickers])
    #
    # # === Simulation Monte Carlo (log-normal process)
    # mean_r = mean_returns.values
    # cov = cov_matrix.values
    #
    # sim_results = np.zeros((n_simulations, years + 1))
    # sim_results[:, 0] = initial_capital
    #
    # for i in range(n_simulations):
    #     port_val = initial_capital
    #     for y in range(1, years + 1):
    #         rand = np.random.multivariate_normal(mean_r, cov)
    #         yearly_return = np.dot(weights, rand)
    #         port_val *= (1 + yearly_return)
    #         sim_results[i, y] = port_val
    # === Paramètres de simulation
    alloc = st.session_state.etf_allocation
    weights = np.array([alloc[ticker] for ticker in etf_tickers])

    mean_r = mean_returns.values  # Rendements annuels
    cov = cov_matrix.values  # Matrice de covariance annuelle

    # === Générer la grille temporelle journalière
    #dates_existantes = real_portfolio.index
    nb_days = nb_month_projection*30 #len(dates_existantes)
    n_simulations = 500

    # # Convertir les rendements et covariances en version journalière
    # trading_days = 252
    # mean_daily = mean_r / trading_days
    # cov_daily = cov / trading_days
    #
    # # === Simulation Monte Carlo (log-normal process, daily steps)
    # sim_results = np.zeros((n_simulations, nb_days))
    # sim_results[:, 0] = initial_capital
    #
    # for i in range(n_simulations):
    #     port_val = initial_capital
    #     for t in range(1, nb_days):
    #         rand = np.random.multivariate_normal(mean_daily, cov_daily)
    #         daily_return = np.dot(weights, rand)
    #         port_val *= (1 + daily_return)
    #         sim_results[i, t] = port_val

    # === Paramètres
    nb_days = nb_month_projection * 30  # Nombre de jours à projeter
    n_simulations = 500
    trading_days = 252

    # Rendements et covariance journalier
    mean_daily = mean_r / trading_days
    cov_daily = cov / trading_days

    # === Générer tous les rendements aléatoires d'un seul coup
    # Shape : (n_simulations, nb_days, n_assets)
    rand_returns = np.random.multivariate_normal(mean_daily, cov_daily, size=(n_simulations, nb_days))

    # === Appliquer les poids (produit matriciel)
    # Shape : (n_simulations, nb_days)
    weighted_returns = np.einsum('ijk,k->ij', rand_returns, weights)

    # === Calcul de la valeur du portefeuille jour après jour
    # On calcule les rendements cumulés : (1 + r1) * (1 + r2) * ...
    cumulative_returns = np.cumprod(1 + weighted_returns, axis=1)

    # === Appliquer le capital initial
    sim_results = initial_capital * cumulative_returns

    # === Calcul des trajectoires statistiques (quantiles journaliers)
    median_sim = np.median(sim_results, axis=0)
    q25_sim = np.percentile(sim_results, 25, axis=0)
    q75_sim = np.percentile(sim_results, 75, axis=0)

    # === Calcul des quantiles
    quantiles = np.percentile(sim_results[:, -1], [25, 50, 75])

    #st.subheader(f"📈 Simulation Monte Carlo du portefeuille retenu ({int(nb_month_projection/12)} Année(s))")
    #st.markdown(f"🔹 **10% quantile** : `{quantiles[0]:,.0f} €`")
    # st.markdown(f"🔹 **25% quantile** : `{quantiles[0]:,.0f} €`")
    # st.markdown(f"🔸 **50% (médiane)** : `{quantiles[1]:,.0f} €`")
    # st.markdown(f"🔹 **75% quantile** : `{quantiles[2]:,.0f} €`")
    #st.markdown(f"🔹 **90% quantile** : `{quantiles[4]:,.0f} €`")
    #st.markdown(f"🔹 **75% quantile** : `{quantiles[5]:,.0f} €`")

    # === Affichage graphique
    fig_sim = go.Figure()
    for i in range(min(n_simulations, 100)):
        fig_sim.add_trace(go.Scatter(
            x=list(range(years + 1)),
            y=sim_results[i],
            mode="lines",
            line=dict(width=0.8, color="gray"),
            showlegend=False,
            opacity=0.2
        ))
    # Ajouter quantile 50%
    fig_sim.add_trace(go.Scatter(
        x=list(range(years + 1)),
        y=np.median(sim_results, axis=0),
        mode="lines",
        line=dict(width=3, color="blue"),
        name="Médiane"
    ))
    # st.plotly_chart(fig_sim, use_container_width=True)

    # === Comparaison avec données réelles sur 20 ans a venir
    st.subheader(f"📊 Comparaison avec l'évolution réelle/projetée sur {int(nb_month_projection/12) if int(nb_month_projection/12)>0 else nb_month_projection} an(s)")
    #full_start = (end_date + dt.timedelta(days=20 * 365)).strftime("%Y-%m-%d")
    full_start = (end_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")


    @st.cache_data
    def load_long_data(tickers, start, end):
        return yf.download(tickers, start=start, end=end)["Close"].dropna()


    long_data = load_long_data(etf_tickers, full_start, (end_date + dt.timedelta(days=nb_month_projection*30)).strftime("%Y-%m-%d"))

    # Normaliser et calculer la valeur du portefeuille dans le passé
    rets_real = long_data.pct_change().dropna()
    weighted_returns = rets_real.dot(weights)
    real_portfolio = (1 + weighted_returns).cumprod() * initial_capital


    # === Portefeuille réel sur la même période (20 ans suivants)
    real_portfolio = (1 + weighted_returns).cumprod() * initial_capital
    dt_end=(end_date + dt.timedelta(days=30 * nb_month_projection)).strftime("%Y-%m-%d")
    # === Tracé de l'évolution réelle
    fig_real = go.Figure()
    fig_real.add_trace(go.Scatter(
        x=real_portfolio.index,
        y=real_portfolio.values,
        mode="lines",
        name=f"Évolution réelle entre {full_start} et {dt_end}",
        line=dict(color="green", width=3)
    ))
    # st.plotly_chart(fig_real, use_container_width=True)

    # === Valeur finale et comparaison aux quantiles
    valeur_finale_reelle = real_portfolio.values[-1]

    ###ici
    # === Calcul des trajectoires médianes / quantiles de la simulation
    median_sim = np.median(sim_results, axis=0)
    q25_sim = np.percentile(sim_results, 25, axis=0)
    q75_sim = np.percentile(sim_results, 75, axis=0)

    # === Dates pour la simulation
    #dates_sim = pd.date_range(start=real_portfolio.index[0], periods=years + 1, freq="Y")

    # 1. Dates réelles connues
    dates_existantes = real_portfolio.index
    end_date = dates_existantes[-1]

    # 2. Nombre de points dans la simulation
    nb_points_sim = len(median_sim)

    # 3. Compléter les dates si la simulation est plus longue
    if nb_points_sim > len(dates_existantes):
        nb_dates_a_ajouter = nb_points_sim - len(dates_existantes)
        dates_a_ajouter = pd.bdate_range(start=end_date + pd.Timedelta(days=1), periods=nb_dates_a_ajouter)
        dates_sim = dates_existantes.append(dates_a_ajouter)
    else:
        dates_sim = dates_existantes[:nb_points_sim]

    # ✅ S'assurer que les dates sont bien triées et sans doublon
    dates_sim = pd.DatetimeIndex(dates_sim).sort_values().unique()

    # ✅ Aligner les tailles
    median_sim = median_sim[:len(dates_sim)]
    q25_sim = q25_sim[:len(dates_sim)]
    q75_sim = q75_sim[:len(dates_sim)]

    # ✅ Récupérer les valeurs du portefeuille réel
    if isinstance(real_portfolio, pd.DataFrame):
        real_values = real_portfolio.iloc[:, 0].values
    else:
        real_values = real_portfolio.values

    # === Affichage
    fig_combined = go.Figure()

    # 1. Historique réel
    fig_combined.add_trace(go.Scatter(
        x=dates_existantes,
        y=real_values,
        mode="lines",
        name="📈 Évolution réelle",
        line=dict(color="green", width=3)
    ))

    # 2. Médiane simulée
    fig_combined.add_trace(go.Scatter(
        x=dates_sim,
        y=median_sim,
        mode="lines",
        name="🔵 Simulation - Médiane",
        line=dict(color="blue", width=2)
    ))

    # 3. Quantile 25%
    fig_combined.add_trace(go.Scatter(
        x=dates_sim,
        y=q25_sim,
        mode="lines",
        name="🟠 Simulation - Quantile 25%",
        line=dict(color="orange", width=2, dash="dash")
    ))

    # 4. Quantile 75%
    fig_combined.add_trace(go.Scatter(
        x=dates_sim,
        y=q75_sim,
        mode="lines",
        name="🟣 Simulation - Quantile 75%",
        line=dict(color="purple", width=2, dash="dash")
    ))

    # === Layout final
    try:
        fig_combined.update_layout(
            title=title_statment,
            xaxis_title="Date",
            yaxis_title="Valeur du portefeuille (€)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            height=600
        )
    except Exception:
        pass

    st.plotly_chart(fig_combined, use_container_width=True)
    # st.write("real_portfolio.shape:", real_portfolio.shape)
    # st.write("median_sim.shape:", np.shape(median_sim))
    # st.write("dates_sim.shape:", dates_sim.shape)
    # st.write("Dernière date réelle:", real_portfolio.index[-1])
    # st.write("Première date simulée:", dates_sim[0])

    ##fin

    st.subheader("📊 Comparaison avec les quantiles simulés")
    st.markdown(f"📅 **Valeur finale réelle du portefeuille** : `{valeur_finale_reelle:,.0f} €`")
    st.markdown(f"🔹 **Quantile 25 %** : `{quantiles[0]:,.0f} €`")
    st.markdown(f"🔸 **Médiane (50 %)** : `{quantiles[1]:,.0f} €`")
    st.markdown(f"🔹 **Quantile 75 %** : `{quantiles[2]:,.0f} €`")

    # === Position réelle par rapport aux quantiles
    if valeur_finale_reelle < quantiles[0]:
        niveau = "📉 Sous le 25e percentile (résultat décevant)"
    elif valeur_finale_reelle < quantiles[1]:
        niveau = "🟠 Entre le 25e et le 50e percentile"
    elif valeur_finale_reelle < quantiles[2]:
        niveau = "🟡 Entre le 50e et le 75e percentile (bon résultat)"
    else:
        niveau = "🟢 Supérieur au 75e percentile (très bon résultat)"

    st.markdown(f"➡️ **Position de la performance réelle :** {niveau}")


else:
    st.info("👉 Veuillez d’abord générer et enregistrer une allocation ETF pour voir la simulation.")

#################################################################################################
from scipy.optimize import minimize

try :
    st.header("📐 Optimisation Globale de patrimoine (Frontière efficiente & contraintes)")

    actifs_opt = ["Livrets", "Immobilier", "SCPI", "Bourse", "Crypto","Participation", "Others"]

    # === Données simulées ===
    returns_simulated = df[actifs_opt].replace(0, np.nan).pct_change().dropna()
    mean_returns = returns_simulated.mean().values
    cov_matrix = returns_simulated.cov().values
    nb_assets = len(actifs_opt)

    # === Simulation de 10 000 portefeuilles aléatoires ===
    n_portfolios = 10000
    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(nb_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol

        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = sharpe
        weights_record.append(weights)

    # DataFrame simulation
    results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe"])
    for i, name in enumerate(actifs_opt):
        results_df[name] = [w[i] for w in weights_record]

    # === Portefeuille optimal (brut - max Sharpe) ===
    max_sharpe_idx = results_df["Sharpe"].idxmax()
    opt_sharpe = results_df.iloc[max_sharpe_idx]

    ############################################
    # === Portefeuille optimal avec contraintes ===
    def negative_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol if port_vol != 0 else np.inf

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x[actifs_opt.index("Immobilier")] - 0.20},
        {"type": "ineq", "fun": lambda x: x[actifs_opt.index("Crypto")]-0.05},
        {"type": "ineq", "fun": lambda x: 0.05-x[actifs_opt.index("Crypto")]},
        {"type": "ineq", "fun": lambda x: x[actifs_opt.index("Bourse")] - 0.20},
        {"type": "ineq", "fun": lambda x: x[actifs_opt.index("Livrets")]-0.05},
        {"type": "ineq", "fun": lambda x: 0.07 - x[actifs_opt.index("Others")]},
    {"type": "ineq", "fun": lambda x: x[actifs_opt.index("Participation")]-0.20},
    {"type": "ineq", "fun": lambda x:  x[actifs_opt.index("SCPI")]-0.05}
    )
    bounds = tuple((0, 1) for _ in range(nb_assets))
    x0 = np.array([1.0/nb_assets] * nb_assets)

    opt_result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # === Visualisation combinée ===
    fig_comb = px.scatter(results_df, x="Volatility", y="Return", color="Sharpe",
                          hover_data=actifs_opt, title="💹 Frontière efficiente (simulée) + Optimisation")

    # Portefeuille optimal (brut)
    fig_comb.add_trace(go.Scatter(
        x=[opt_sharpe["Volatility"]],
        y=[opt_sharpe["Return"]],
        mode="markers+text",
        marker=dict(color='red', size=12, symbol="star"),
        name="Max Sharpe (sans contraintes)",
        text=["Max Sharpe"],
        textposition="top center"
    ))

    # Portefeuille optimal avec contraintes
    if opt_result.success:
        w_opt = opt_result.x
        r_opt = np.dot(w_opt, mean_returns)
        v_opt = np.sqrt(np.dot(w_opt.T, np.dot(cov_matrix, w_opt)))
        s_opt = r_opt / v_opt

        fig_comb.add_trace(go.Scatter(
            x=[v_opt],
            y=[r_opt],
            mode="markers+text",
            marker=dict(color='blue', size=12, symbol="diamond"),
            name="Max Sharpe (avec contraintes)",
            text=["Contraint"],
            textposition="bottom center"
        ))
        st.plotly_chart(fig_comb, use_container_width=True)

        st.subheader("🔒 Allocation optimale (avec contraintes)")
        for i, a in enumerate(actifs_opt):
            st.markdown(f"- **{a}** : {w_opt[i]*100:.1f}%")
        st.markdown(f"📈 **Rendement attendu** : `{r_opt*100:.2f} %`")
        st.markdown(f"📉 **Volatilité estimée** : `{v_opt*100:.2f} %`")
        st.markdown(f"📊 **Ratio de Sharpe** : `{s_opt:.2f}`")
    else:
        st.error("❌ Optimisation avec contraintes échouée.")

    # Résumé du portefeuille max Sharpe brut
    st.subheader("💡 Allocation max Sharpe (sans contraintes)")
    for a in actifs_opt:
        st.markdown(f"- **{a}** : {opt_sharpe[a]*100:.1f}%")
    st.markdown(f"📈 **Rendement attendu** : `{opt_sharpe['Return']*100:.2f} %`")
    st.markdown(f"📉 **Volatilité estimée** : `{opt_sharpe['Volatility']*100:.2f} %`")
    st.markdown(f"📊 **Ratio de Sharpe** : `{opt_sharpe['Sharpe']:.2f}`")
except Exception:
    pass
    st.write ("Certaines informations sont manquantes pour effectuer l'optimisation globale, vérifier les données insérées en amont!")



