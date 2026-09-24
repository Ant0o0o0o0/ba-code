from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from soep_preparation.config import BLD, MODULES
from soep_preparation.final_dataset import create_final_dataset

# Die vier Wellen mit erhobenem Vermögen.
SURVEY_YEARS = [2002, 2007, 2012, 2017]

# Die fünf Imputationen des SOEP.
IMPLICATES = ["a", "b", "c", "d", "e"]

# Vermögensmaße auf Haushaltsebene, jeweils über alle fünf Imputationen.
# net_overall_wealth      -> Nettovermögen (entspricht Carrolls "net worth")
# financial_assets_value  -> Finanzvermögen (Näherung an "liquid assets")
WEALTH_VARIABLES = [
    f"hh_{stem}_{impl}"
    for stem in ("net_overall_wealth", "financial_assets_value")
    for impl in IMPLICATES
]

VARIABLES_TO_MERGE = [
    # Index
    "survey_year",
    "hh_id",
    "p_id",
    # Gewichtung: ohne Haushaltsgewicht sind die Lorenz-Anteile nicht
    # bevölkerungsrepräsentativ und damit nicht mit den Zielwerten
    # bei Carroll et al. vergleichbar.
    "hh_weighting_factor",
    # Regionale Zuordnung für den Ost/West-Vergleich.
    "federal_state_of_residence",
    # Nenner der Vermögens-Einkommens-Quote.
    "income_after_tax_y_hh",
    # Erlaubt, genau eine Zeile pro Haushalt auszuwählen.
    "relationship_to_head_of_hh",
    # Deskriptive Merkmale des Haushaltsvorstands.
    "age",
    "employment_status",
    *WEALTH_VARIABLES,
]


@task(after="create_metadata")
def task_ba_extract_wealth(
    modules: Annotated[dict[str, pd.DataFrame], MODULES._entries],  # noqa: SLF001
    variables: Annotated[list[str], VARIABLES_TO_MERGE],
    survey_years: Annotated[list[int], SURVEY_YEARS],
    out_path: Annotated[Path, Product] = BLD / "ba_mpc" / "wealth_extract.pickle",
) -> None:
    """Extrahiert die Variablen für die MPC-Kalibrierung.

    Args:
        modules: Alle aufbereiteten Module der Pipeline.
        variables: Die zu extrahierenden Variablennamen.
        survey_years: Die Erhebungsjahre mit Vermögensinformationen.
        out_path: Zielpfad des extrahierten Datensatzes.

    Returns:
        None
    """
    dataset = create_final_dataset(
        modules=modules,
        variables=variables,
        survey_years=survey_years,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pickle(out_path)
