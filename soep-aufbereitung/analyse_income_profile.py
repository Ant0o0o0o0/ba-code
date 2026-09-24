from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

IN_PATH = "bld/ba_mpc/income_profile_extract.pickle"
OUT_PATH = "bld/ba_mpc/income_profile.csv"

# Altersspanne der Schätzung. Unter 25 sind zu viele Haushalte in Ausbildung,
# über 80 wird die Stichprobe dünn.
AGE_MIN = 25
AGE_MAX = 80

# Modellannahmen zum Erwerbsverlauf.
AGE_START_WORKING = 25
AGE_RETIREMENT = 67

# Für die Ersatzrate beim Renteneintritt verglichene Altersfenster.
PRE_RETIREMENT_AGES = (60, 66)
POST_RETIREMENT_AGES = (68, 75)

# Grad des Polynoms für die geglättete Profilvariante.
SMOOTHING_DEGREE = 4

EAST_STATES = [
    "Brandenburg",
    "Mecklenburg-Vorpommern",
    "Saxony",
    "Saxony-Anhalt",
    "Thuringia",
]

# Mindestzahl an Beobachtungen je Altersjahr, damit ein Alter berücksichtigt
# wird.
MIN_OBS_PER_AGE = 30

CHUNK_SIZE = 50_000


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------


def as_mask(condition: pd.Series) -> np.ndarray:
    """Wandelt einen Vergleich in eine Maske ohne fehlende Werte um.

    Args:
        condition: Ergebnis eines Vergleichs, ggf. mit fehlenden Werten.

    Returns:
        Boolescher Array ohne fehlende Werte.
    """
    return condition.fillna(value=False).to_numpy(dtype=bool)


def numeric(series: pd.Series) -> np.ndarray:
    """Wandelt eine Spalte in einen float-Array um.

    Args:
        series: Eingabespalte, ggf. mit pyarrow-Datentyp.

    Returns:
        Werte als float-Array, fehlende Werte als NaN.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()


def to_household_level(data: pd.DataFrame) -> pd.DataFrame:
    """Reduziert den Datensatz auf eine Zeile pro Haushalt und Welle.

    Args:
        data: Datensatz auf Personenebene.

    Returns:
        Datensatz auf Haushaltsebene, bevorzugt die Zeile des Vorstands.
    """
    is_head = as_mask(
        data["relationship_to_head_of_hh"].astype("string") == "Household head"
    )
    data = data.assign(_head_first=np.where(is_head, 0, 1))
    data = data.sort_values(["survey_year", "hh_id", "_head_first"])
    return (
        data.drop_duplicates(subset=["survey_year", "hh_id"], keep="first")
        .drop(columns="_head_first")
        .reset_index(drop=True)
    )


def assign_region(data: pd.DataFrame) -> pd.DataFrame:
    """Ordnet jedem Haushalt eine Region zu.

    Args:
        data: Datensatz mit `federal_state_of_residence`.

    Returns:
        Datensatz mit zusätzlicher Spalte `region`.
    """
    state = data["federal_state_of_residence"].astype("string")
    region = pd.Series("West", index=data.index, dtype="object")
    region[as_mask(state.isin(EAST_STATES))] = "East"
    region[as_mask(state == "Berlin")] = "Berlin"
    region[state.isna().to_numpy(dtype=bool)] = pd.NA
    return data.assign(region=region)


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Bereitet den Datensatz für die Regression vor.

    Erzeugt das äquivalisierte Einkommen, korrigiert den Altersbezug und
    beschränkt auf verwertbare Beobachtungen.

    Args:
        data: Extrahierter Datensatz auf Personenebene.

    Returns:
        Datensatz auf Haushaltsebene mit den Spalten für die Regression.
    """
    data = to_household_level(data)
    data = assign_region(data)

    income = numeric(data["income_after_tax_y_hh"])
    size = numeric(data["number_of_persons_hh"])
    weight = numeric(data["hh_weighting_factor"])
    age = numeric(data["age"])
    year = numeric(data["survey_year"])

    # Bedarfsgewichtung nach der Quadratwurzelskala. Das Modell kennt keine
    # Haushaltsgröße; ohne Gewichtung würde das Altersprofil die Veränderung
    # der Haushaltsgröße über den Lebenszyklus mit abbilden.
    with np.errstate(invalid="ignore", divide="ignore"):
        equivalised = income / np.sqrt(size)

    # Das Einkommen bezieht sich auf das Vorjahr, das Alter auf den
    # Erhebungszeitpunkt. Beides wird auf das Einkommensjahr bezogen.
    age_at_income = age - 1.0
    income_year = year - 1.0

    out = data.assign(
        _equivalised=equivalised,
        _weight=weight,
        _age=age_at_income,
        _income_year=income_year,
    )

    keep = (
        (out["_equivalised"].to_numpy() > 0)
        & np.isfinite(out["_equivalised"].to_numpy())
        & (out["_weight"].to_numpy() > 0)
        & np.isfinite(out["_weight"].to_numpy())
        & (out["_age"].to_numpy() >= AGE_MIN)
        & (out["_age"].to_numpy() <= AGE_MAX)
    )
    return out[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Gewichtete Regression mit Dummy-Regressoren
# --------------------------------------------------------------------------


def fit_weighted_dummies(
    log_income: np.ndarray,
    age: np.ndarray,
    year: np.ndarray,
    weight: np.ndarray,
) -> dict[int, float]:
    """Schätzt Alterseffekte in einer gewichteten Regression mit Jahresdummies.

    Die Normalgleichungen werden blockweise akkumuliert, damit die
    Designmatrix nicht vollständig im Speicher liegen muss.

    Args:
        log_income: Logarithmiertes äquivalisiertes Einkommen.
        age: Alter im Einkommensjahr.
        year: Einkommensjahr.
        weight: Hochrechnungsfaktor.

    Returns:
        Alterseffekte im Logarithmus, normiert auf das jüngste Alter.
    """
    ages = np.unique(age)
    years = np.unique(year)

    # Basiskategorien: jüngstes Alter und erstes Jahr entfallen, dafür gibt es
    # eine Konstante.
    age_index = {a: i for i, a in enumerate(ages[1:])}
    year_index = {y: i for i, y in enumerate(years[1:])}
    n_age = len(age_index)
    n_year = len(year_index)
    n_col = 1 + n_age + n_year

    xtx = np.zeros((n_col, n_col))
    xty = np.zeros(n_col)

    for start in range(0, len(log_income), CHUNK_SIZE):
        end = start + CHUNK_SIZE
        a_chunk = age[start:end]
        y_chunk = year[start:end]
        w_chunk = weight[start:end]
        v_chunk = log_income[start:end]

        design = np.zeros((len(a_chunk), n_col))
        design[:, 0] = 1.0
        for pos, value in enumerate(a_chunk):
            if value in age_index:
                design[pos, 1 + age_index[value]] = 1.0
        for pos, value in enumerate(y_chunk):
            if value in year_index:
                design[pos, 1 + n_age + year_index[value]] = 1.0

        weighted = design * w_chunk[:, None]
        xtx += design.T @ weighted
        xty += weighted.T @ v_chunk

    coefficients = np.linalg.solve(xtx, xty)

    effects = {int(ages[0]): 0.0}
    for value, i in age_index.items():
        effects[int(value)] = float(coefficients[1 + i])
    return effects


# --------------------------------------------------------------------------
# Profil und Wachstumsfaktoren
# --------------------------------------------------------------------------


def build_profile(effects: dict[int, float]) -> pd.DataFrame:
    """Baut aus den Alterseffekten das Niveauprofil und die Wachstumsfaktoren.

    Args:
        effects: Alterseffekte im Logarithmus je Alter.

    Returns:
        Tabelle mit Niveau, Gamma und geglätteten Varianten je Alter.
    """
    ages = np.array(sorted(effects))
    log_level = np.array([effects[int(a)] for a in ages])

    # Auf das jüngste Alter normieren.
    log_level = log_level - log_level[0]
    level = np.exp(log_level)

    # Geglättete Variante: Polynom im Alter, getrennt für Erwerbsphase und
    # Rentenphase, damit der Bruch beim Renteneintritt nicht verschliffen wird.
    smooth_log = np.full_like(log_level, np.nan)
    for mask in (ages < AGE_RETIREMENT, ages >= AGE_RETIREMENT):
        if mask.sum() > SMOOTHING_DEGREE + 1:
            fit = np.polyfit(ages[mask], log_level[mask], SMOOTHING_DEGREE)
            smooth_log[mask] = np.polyval(fit, ages[mask])
        elif mask.sum() > 0:
            smooth_log[mask] = log_level[mask]
    smooth_level = np.exp(smooth_log)

    # Gamma(Alter) ist der Wachstumsfaktor von Alter zu Alter+1.
    gamma = np.full_like(level, np.nan)
    gamma[:-1] = level[1:] / level[:-1]
    gamma_smooth = np.full_like(smooth_level, np.nan)
    gamma_smooth[:-1] = smooth_level[1:] / smooth_level[:-1]

    return pd.DataFrame(
        {
            "age": ages,
            "level": level,
            "level_smooth": smooth_level,
            "gamma": gamma,
            "gamma_smooth": gamma_smooth,
        }
    )


def replacement_rate(profile: pd.DataFrame) -> float:
    """Berechnet die Einkommens-Ersatzrate beim Renteneintritt.

    Args:
        profile: Tabelle mit Niveauprofil je Alter.

    Returns:
        Verhältnis des Niveaus nach zum Niveau vor dem Renteneintritt.
    """
    age = profile["age"].to_numpy()
    level = profile["level"].to_numpy()

    before = (age >= PRE_RETIREMENT_AGES[0]) & (age <= PRE_RETIREMENT_AGES[1])
    after = (age >= POST_RETIREMENT_AGES[0]) & (age <= POST_RETIREMENT_AGES[1])
    if before.sum() == 0 or after.sum() == 0:
        return np.nan
    return float(np.nanmean(level[after]) / np.nanmean(level[before]))


# --------------------------------------------------------------------------
# Ablauf
# --------------------------------------------------------------------------


def analyse_group(group: pd.DataFrame, label: str) -> pd.DataFrame | None:
    """Schätzt das Altersprofil für eine Region.

    Args:
        group: Haushalte einer Region über alle Wellen.
        label: Bezeichnung der Region für die Ausgabe.

    Returns:
        Profiltabelle oder None, wenn zu wenige Beobachtungen vorliegen.
    """
    # Altersjahre mit zu wenigen Beobachtungen entfernen, damit einzelne
    # Dummies nicht auf wenigen Fällen beruhen.
    counts = group["_age"].value_counts()
    valid_ages = counts[counts >= MIN_OBS_PER_AGE].index
    group = group[group["_age"].isin(valid_ages)]

    if len(group) < 1000:  # noqa: PLR2004
        print(f"  {label}: zu wenige Beobachtungen ({len(group)}), übersprungen")
        return None

    log_income = np.log(group["_equivalised"].to_numpy())
    effects = fit_weighted_dummies(
        log_income=log_income,
        age=group["_age"].to_numpy(),
        year=group["_income_year"].to_numpy(),
        weight=group["_weight"].to_numpy(),
    )

    profile = build_profile(effects)
    profile.insert(0, "region", label)

    rate = replacement_rate(profile)
    working = profile[
        (profile["age"] >= AGE_START_WORKING) & (profile["age"] < AGE_RETIREMENT)
    ]
    mean_growth = float(np.nanmean(working["gamma_smooth"]))
    peak_age = int(working.loc[working["level_smooth"].idxmax(), "age"])

    print(
        f"  {label}: n={len(group)}, "
        f"Gamma im Mittel (Erwerbsphase)={mean_growth:.4f}, "
        f"Maximum bei Alter {peak_age}, "
        f"Ersatzrate Rente={rate:.3f}"
    )
    return profile


def main() -> None:
    """Führt die Schätzung aus und schreibt die Profiltabelle."""
    data = pd.read_pickle(IN_PATH)  # noqa: S301
    data = prepare(data)

    print(f"Verwertbare Haushaltsbeobachtungen: {len(data)}")
    print(
        "Einkommensjahre: "
        f"{int(data['_income_year'].min())}-{int(data['_income_year'].max())}\n"
    )
    print("Schätzung des Altersprofils:")

    groups: list[tuple[str, pd.DataFrame]] = [("Germany", data)]
    groups += [
        (str(name), grp)
        for name, grp in data.groupby("region", dropna=True, observed=True)
        if str(name) in ("East", "West")
    ]

    profiles = [
        profile
        for label, group in groups
        if (profile := analyse_group(group, label)) is not None
    ]

    out = pd.concat(profiles, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nGeschrieben: {OUT_PATH}")
    print("\nProfil für Deutschland (jedes fünfte Alter):")
    germany = out[out["region"] == "Germany"]
    print(germany[germany["age"] % 5 == 0].to_string(index=False))


if __name__ == "__main__":
    main()
