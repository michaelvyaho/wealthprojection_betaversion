import numpy_financial as npf
import pandas as pd


def generer_tableau_amortissement(
    montant_emprunte,
    taux_annuel,
    duree_annees,
    annee_debut,
    annee_fin=None,
    appreciation=0.01
):
    """
    Génère un tableau d'amortissement annuel avec calcul du patrimoine net.

    Args:
        montant_emprunte (float): Montant du prêt.
        taux_annuel (float): Taux d’intérêt annuel (ex: 0.02 pour 2%).
        duree_annees (int): Durée du prêt en années.
        annee_debut (int): Année de démarrage du prêt.
        annee_fin (int): Année de fin de projection. Si None, calcule jusqu'à annee_debut + 90.
        appreciation (float): Taux d'appréciation annuelle du bien.

    Returns:
        pd.DataFrame: Tableau annuel avec valeur estimée du bien et patrimoine net.
    """
    if annee_fin is None:
        annee_fin = annee_debut + 90

    mensualite = -npf.pmt(taux_annuel / 12, duree_annees * 12, montant_emprunte)
    capital_restant = montant_emprunte
    data = []

    for i, annee in enumerate(range(annee_debut, annee_fin + 1)):
        interets_annuels = 0
        principal_annuel = 0

        if i < duree_annees:
            for mois in range(12):
                interet_mensuel = capital_restant * (taux_annuel / 12)
                principal_mensuel = mensualite - interet_mensuel
                capital_restant -= principal_mensuel

                interets_annuels += interet_mensuel
                principal_annuel += principal_mensuel

            annuite_totale = mensualite * 12
        else:
            annuite_totale = 0  # plus d'annuité après fin du prêt

        valeur_bien = montant_emprunte * ((1 + appreciation) ** i)
        patrimoine_net = valeur_bien - max(capital_restant, 0)

        data.append({
            "Année": annee,
            "Capital restant dû": max(capital_restant, 0),
            "Intérêts payés": interets_annuels,
            "Principal remboursé": principal_annuel,
            "Annuité totale": annuite_totale,
            "Valeur estimée bien": valeur_bien,
            "PatrimoineNet": patrimoine_net
        })

    df = pd.DataFrame(data).set_index("Année")
    return df


def generer_tableau_amortissement_old(montant_emprunte, taux_annuel, duree_annees, annee_debut, appreciation=0.01):
    """
    Génère un tableau d'amortissement annuel avec calcul du patrimoine net.

    Args:
        montant_emprunte (float): Montant du prêt.
        taux_annuel (float): Taux d’intérêt annuel (ex: 0.02 pour 2%).
        duree_annees (int): Durée du prêt en années.
        annee_debut (int): Année de démarrage du prêt.
        appreciation (float): Taux d'appréciation annuelle de la valeur du bien.

    Returns:
        pd.DataFrame: Tableau d'amortissement avec le patrimoine net.
    """
    mensualite = -npf.pmt(taux_annuel / 12, duree_annees * 12, montant_emprunte)
    capital_restant = montant_emprunte
    data = []

    for i in range(duree_annees):
        annee = annee_debut + i
        interets_annuels = 0
        principal_annuel = 0

        for mois in range(12):
            interet_mensuel = capital_restant * (taux_annuel / 12)
            principal_mensuel = mensualite - interet_mensuel
            capital_restant -= principal_mensuel

            interets_annuels += interet_mensuel
            principal_annuel += principal_mensuel

        valeur_bien = montant_emprunte * ((1 + appreciation) ** i)
        patrimoine_net = valeur_bien - max(capital_restant, 0)

        data.append({
            "Année": annee,
            "Capital restant dû": max(capital_restant, 0),
            "Intérêts payés": interets_annuels,
            "Principal remboursé": principal_annuel,
            "Annuité totale": mensualite * 12,
            "Valeur estimée bien": valeur_bien,
            "PatrimoineNet": patrimoine_net
        })



    df = pd.DataFrame(data).set_index("Année")
    return df

if __name__ == '__main__':
    montant_emprunte=236000
    taux_annuel=2/100
    duree_annees=25
    annee_debut=2022

    res=generer_tableau_amortissement(montant_emprunte, taux_annuel, duree_annees, annee_debut, appreciation=0.02)
    print(res.head(1))