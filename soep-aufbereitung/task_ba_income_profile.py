from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from soep_preparation.config import BLD, MODULES
from soep_preparation.final_dataset import create_final_dataset

SURVEY_YEARS = list(range(1992, 2020))

VARIABLES_TO_MERGE = [
    # Index
    "survey_year",
    "hh_id",
    "p_id",
    # Gewichtung
    "hh_weighting_factor",
    # Altersprofil
    "age",
    # Zielgröße
    "income_after_tax_y_hh",
    # Für die Äquivalenzgewichtung: das Modell kennt keine Haushaltsgröße,
    # deshalb wird das Einkommen bedarfsgewichtet.
    "number_of_persons_hh",
    # Regionale Zuordnung
    "federal_state_of_residence",
    # Auswahl genau einer Zeile pro Haushalt
    "relationship_to_head_of_hh",
    # Erlaubt, den Renteneintritt im Profil zu erkennen.
    "employment_status",
]


@task(after="create_metadata")
def task_ba_extract_income_profile(
    modules: Annotated[dict[str, pd.DataFrame], MODULES._entries],  # noqa: SLF001
    variables: Annotated[list[str], VARIABLES_TO_MERGE],
    survey_years: Annotated[list[int], SURVEY_YEARS],
    out_path: Annotated[Path, Product] = BLD
    / "ba_mpc"
    / "income_profile_extract.pickle",
) -> None:
    """Extrahiert Alter und Haushaltseinkommen über alle relevanten Wellen.

    Args:
        modules: Alle aufbereiteten Module der Pipeline.
        variables: Die zu extrahierenden Variablennamen.
        survey_years: Die zu berücksichtigenden Erhebungsjahre.
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
