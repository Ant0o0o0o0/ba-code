from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

IN_PATH = "bld/ba_mpc/wealth_extract.pickle"
OUT_PATH = "bld/ba_mpc/calibration_targets.csv"

# Hauptwelle der Analyse. 2017 ist die aktuellste Welle mit beobachtetem
# Vermögen in SOEP-Core V41. 2012 eignet sich als Robustheitscheck.
MAIN_YEAR = 2017

# Zielperzentile: die vier von Carroll et al. angepassten Momente (20/40/60/80)
# plus 1 und 10 als nicht angepasste Kontrollgrößen.
TOP_SHARES = [1, 10, 20, 40, 60, 80]

IMPLICATES = ["a", "b", "c", "d", "e"]

WEALTH_MEASURES = {
    "net_worth": "hh_net_overall_wealth",
    "financial_assets": "hh_financial_assets_value",
}

# Die fünf ostdeutschen Flächenländer. Berlin wird bewusst getrennt geführt:
# die Stadt ist weder eindeutig Ost noch West, und ihre Zuordnung verschiebt
# die Ergebnisse messbar. Die Entscheidung gehört in den Methodenteil.
EAST_STATES = [
    "Brandenburg",
    "Mecklenburg-Vorpommern",
    "Saxony",
    "Saxony-Anhalt",
    "Thuringia",
]

N_BOOTSTRAP = 200
RNG_SEED = 20260811


# --------------------------------------------------------------------------
# Kennzahlen
# --------------------------------------------------------------------------


def weighted_top_share(
    wealth: np.ndarray,
    weight: np.ndarray,
    top_pct: float,
) -> float:
    """Anteil am Gesamtvermögen, der auf die reichsten `top_pct` Prozent entfällt.

    Die Haushalte werden nach Vermögen absteigend sortiert. Am Rand der Gruppe
    wird linear interpoliert, damit das Ergebnis nicht davon abhängt, ob ein
    einzelner Haushalt gerade noch in die Gruppe fällt.

    Args:
        wealth: Vermögen je Haushalt.
        weight: Hochrechnungsfaktor je Haushalt.
        top_pct: Größe der Gruppe in Prozent der Haushalte.

    Returns:
        Anteil am Gesamtvermögen in Prozent.
    """
    order = np.argsort(-wealth, kind="stable")
    w = wealth[order]
    g = weight[order]

    total_weight = g.sum()
    total_wealth = (w * g).sum()
    if total_weight <= 0 or total_wealth == 0:
        return np.nan

    cutoff = total_weight * top_pct / 100.0
    cum_weight = np.cumsum(g)

    # Haushalte, die vollständig in die Gruppe fallen.
    full = cum_weight <= cutoff
    covered = (w[full] * g[full]).sum()
    used_weight = g[full].sum()

    # Der erste nicht vollständig enthaltene Haushalt geht anteilig ein.
    idx = int(full.sum())
    if idx < len(w):
        remaining = cutoff - used_weight
        if remaining > 0:
            covered += w[idx] * remaining

    return 100.0 * covered / total_wealth


def weighted_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Verhältnis zweier gewichteter Aggregate, z. B. Vermögen zu Einkommen.

    Args:
        numerator: Zählergröße je Haushalt.
        denominator: Nennergröße je Haushalt.
        weight: Hochrechnungsfaktor je Haushalt.

    Returns:
        Das Verhältnis der gewichteten Summen.
    """
    denom = (denominator * weight).sum()
    if denom == 0:
        return np.nan
    return (numerator * weight).sum() / denom


# --------------------------------------------------------------------------
# Multiple Imputation
# --------------------------------------------------------------------------


def combine_rubin(estimates: list[float], variances: list[float]) -> dict[str, float]:
    """Fasst Schätzungen aus mehreren Imputationen nach den Rubin-Regeln zusammen.

    Args:
        estimates: Punktschätzer je Imputation.
        variances: Varianz innerhalb je Imputation (z. B. per Bootstrap).

    Returns:
        Punktschätzer, Standardfehler und Anteil der Imputationsvarianz.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)

    q_bar = q.mean()
    u_bar = u.mean()
    # Varianz zwischen den Imputationen.
    b = q.var(ddof=1) if m > 1 else 0.0
    total = u_bar + (1.0 + 1.0 / m) * b

    return {
        "estimate": q_bar,
        "se": np.sqrt(total) if total > 0 else np.nan,
        "share_imputation_variance": (
            (1.0 + 1.0 / m) * b / total if total > 0 else np.nan
        ),
    }


def bootstrap_variance(
    func,
    wealth: np.ndarray,
    weight: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Schätzt die Stichprobenvarianz einer Kennzahl per Bootstrap.

    Args:
        func: Funktion, die aus Vermögen und Gewicht eine Kennzahl berechnet.
        wealth: Vermögen je Haushalt.
        weight: Hochrechnungsfaktor je Haushalt.
        rng: Zufallsgenerator.

    Returns:
        Varianz der Kennzahl über die Bootstrap-Ziehungen.
    """
    n = len(wealth)
    draws = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        pick = rng.integers(0, n, size=n)
        draws[i] = func(wealth[pick], weight[pick])
    return float(np.nanvar(draws, ddof=1))


# --------------------------------------------------------------------------
# Datenaufbereitung
# --------------------------------------------------------------------------


def as_mask(condition: pd.Series) -> np.ndarray:
    """Wandelt einen Vergleich in eine Maske ohne fehlende Werte um.

    Vergleiche auf Spalten mit fehlenden Werten liefern in pandas `pd.NA`
    statt True oder False. Als Maske oder in einer Umwandlung in Zahlen
    führt das zu einem Fehler, deshalb werden fehlende Werte hier als
    "Bedingung nicht erfüllt" behandelt.

    Args:
        condition: Ergebnis eines Vergleichs, ggf. mit fehlenden Werten.

    Returns:
        Boolescher Array ohne fehlende Werte.
    """
    return condition.fillna(value=False).to_numpy(dtype=bool)


def to_household_level(data: pd.DataFrame) -> pd.DataFrame:
    """Reduziert den Datensatz auf eine Zeile pro Haushalt und Welle.

    Bevorzugt wird die Zeile des Haushaltsvorstands, weil Alter und
    Erwerbsstatus dann interpretierbar sind. Fehlt sie, wird die erste
    verfügbare Zeile des Haushalts verwendet.

    Args:
        data: Datensatz auf Personenebene.

    Returns:
        Datensatz auf Haushaltsebene.
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


def numeric(series: pd.Series) -> np.ndarray:
    """Wandelt eine Spalte in einen float-Array um.

    Args:
        series: Eingabespalte, ggf. mit pyarrow-Datentyp.

    Returns:
        Werte als float-Array, fehlende Werte als NaN.
    """
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()


# --------------------------------------------------------------------------
# Auswertung
# --------------------------------------------------------------------------


def targets_for_group(
    group: pd.DataFrame,
    stem: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Berechnet alle Kalibrierungsziele für eine Region und ein Vermögensmaß.

    Args:
        group: Haushalte einer Region in einer Welle.
        stem: Präfix der Vermögensvariable ohne Imputationssuffix.
        rng: Zufallsgenerator für den Bootstrap.

    Returns:
        Eine Liste von Ergebniszeilen.
    """
    rows: list[dict] = []

    # Je Zielmoment über die Imputationen sammeln.
    share_estimates: dict[int, list[float]] = {p: [] for p in TOP_SHARES}
    share_variances: dict[int, list[float]] = {p: [] for p in TOP_SHARES}
    ratio_estimates: list[float] = []
    n_used: list[int] = []

    income = numeric(group["income_after_tax_y_hh"])
    weight_all = numeric(group["hh_weighting_factor"])

    for impl in IMPLICATES:
        wealth_all = numeric(group[f"{stem}_{impl}"])

        # Vollständige Fälle für die Verteilungskennzahlen.
        ok = ~np.isnan(wealth_all) & ~np.isnan(weight_all) & (weight_all > 0)
        wealth = wealth_all[ok]
        weight = weight_all[ok]
        n_used.append(int(ok.sum()))

        if len(wealth) < 30:  # noqa: PLR2004
            continue

        for p in TOP_SHARES:
            share_estimates[p].append(weighted_top_share(wealth, weight, p))
            share_variances[p].append(
                bootstrap_variance(
                    lambda w, g, p=p: weighted_top_share(w, g, p),
                    wealth,
                    weight,
                    rng,
                )
            )

        # Vermögens-Einkommens-Quote: nur Haushalte mit beiden Größen und
        # positivem Einkommen.
        ok_ratio = ok & ~np.isnan(income) & (income > 0)
        if ok_ratio.sum() >= 30:  # noqa: PLR2004
            ratio_estimates.append(
                weighted_ratio(
                    wealth_all[ok_ratio],
                    income[ok_ratio],
                    weight_all[ok_ratio],
                )
            )

    for p in TOP_SHARES:
        if not share_estimates[p]:
            continue
        combined = combine_rubin(share_estimates[p], share_variances[p])
        rows.append(
            {
                "moment": f"top_{p}pct_share",
                "unit": "percent_of_total_wealth",
                "is_calibration_target": p in (20, 40, 60, 80),
                **combined,
            }
        )

    if ratio_estimates:
        rows.append(
            {
                "moment": "wealth_to_income_ratio",
                "unit": "annual_income",
                "is_calibration_target": True,
                "estimate": float(np.mean(ratio_estimates)),
                "se": np.nan,
                "share_imputation_variance": np.nan,
            }
        )

    for row in rows:
        row["n_households"] = int(np.mean(n_used)) if n_used else 0

    return rows


def main() -> None:
    """Führt die Auswertung aus und schreibt die Ergebnistabelle."""
    rng = np.random.default_rng(RNG_SEED)

    data = pd.read_pickle(IN_PATH)  # noqa: S301
    data = data[as_mask(data["survey_year"] == MAIN_YEAR)]
    data = data[as_mask(data["hh_weighting_factor"].notna())]
    """filtert Haushalte ohne Regions angaben"""
    data = to_household_level(data)
    data = assign_region(data)

    print(f"Welle {MAIN_YEAR}: {len(data)} Haushalte\n")
    print("Haushalte je Region:")
    print(data["region"].value_counts(dropna=False).to_string(), "\n")

    results: list[dict] = []

    # "Germany" als Gesamtreferenz, danach die regionale Aufteilung.
    groups: list[tuple[str, pd.DataFrame]] = [("Germany", data)]
    groups += [
        (str(name), grp)
        for name, grp in data.groupby("region", dropna=True, observed=True)
    ]

    for region, group in groups:
        for measure, stem in WEALTH_MEASURES.items():
            for row in targets_for_group(group, stem, rng):
                results.append(
                    {
                        "survey_year": MAIN_YEAR,
                        "region": region,
                        "wealth_measure": measure,
                        **row,
                    }
                )

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)

    print("Kalibrierungsziele:")
    print(
        out[
            [
                "region",
                "wealth_measure",
                "moment",
                "estimate",
                "se",
                "n_households",
            ]
        ].to_string(index=False)
    )
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()