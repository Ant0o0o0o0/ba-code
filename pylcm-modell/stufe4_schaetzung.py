import numpy as np
import pandas as pd

from stufe2_schocks import (
    FIRST_AGE,
    REGION_INPUTS,
    build_model,
    make_params,
    wealth_shares,
)

# Deine SOEP-Ziele (Nettovermoegen), Prozent des Gesamtvermoegens.
SOEP = {
    "Ost":  {"Top 1%": 17.61, "Top 10%": 52.61, "Top 20%": 73.54, "Top 40%": 95.15},
    "West": {"Top 1%": 17.51, "Top 10%": 52.15, "Top 20%": 70.87, "Top 40%": 92.17},
}
# Deine SOEP-Vermoegen/Einkommen (Nettovermoegen) -- nur als Diagnose fuers Niveau.
SOEP_WEALTH_TO_INCOME = {"Ost": 3.02, "West": 5.79}

TARGET_MOMENTS = ("Top 10%", "Top 20%", "Top 40%")  # in die Zielfunktion
DIAG_MOMENT = "Top 1%"                                # nur Diagnose

N_AGENTS = 2000  # fixe Ziehungen (seed) -> deterministische Zielfunktion in beta


def solve_and_summarize(model, gamma, replacement_rate, beta):
    """Loest+simuliert das Modell fuer ein beta und gibt Anteile + V/E zurueck."""
    params = make_params(gamma=gamma, replacement_rate=replacement_rate, beta=beta)

    initial_df = pd.DataFrame(
        {
            "regime_name": "working_life",
            "age": float(FIRST_AGE),
            "wealth": np.linspace(0.0, 2.0, N_AGENTS),
            "z_perm": 0.0,
            "tran_shock": 0.0,
        }
    )
    result = model.simulate(
        params=params,
        initial_conditions=initial_df,
        period_to_regime_to_V_arr=None,
        log_level="off",   # in der Schleife: keine Debug-Ausgaben
        seed=0,
    )
    df = result.to_dataframe(additional_targets="all").dropna(
        subset=["wealth", "consumption"]
    )
    shares = wealth_shares(df["wealth"].to_numpy())
    # grobes Vermoegen/Einkommen: mittleres Vermoegen / mittleres Erwerbseinkommen
    mean_income = df.loc[df["regime_name"] == "working_life", "income"].mean()
    wealth_to_income = float(df["wealth"].mean() / mean_income)
    return shares, wealth_to_income


def objective(shares):
    """Summe der quadrierten Abweichungen auf den Zielmomenten (region-spezifisch
    wird das SOEP-Ziel unten eingesetzt)."""
    # wird in estimate_region mit dem passenden Ziel aufgerufen
    raise NotImplementedError


def estimate_region(model, region, beta_grid):
    gamma = REGION_INPUTS[region]["gamma"]
    repl = REGION_INPUTS[region]["replacement_rate"]
    target = SOEP[region]

    print("=" * 78)
    print(f"REGION {region} -- beta-Profil (Zielfunktion ueber beta)")
    print("=" * 78)
    header = f"{'beta':>7} {'f(beta)':>9} " + " ".join(
        f"{m:>8}" for m in ("Top 1%", "Top 10%", "Top 20%", "Top 40%")
    ) + f" {'V/E':>6}"
    print(header)

    results = []
    for beta in beta_grid:
        shares, wti = solve_and_summarize(model, gamma, repl, beta)
        f = sum((shares[m] - target[m]) ** 2 for m in TARGET_MOMENTS)
        results.append((beta, f, shares, wti))
        line = f"{beta:>7.4f} {f:>9.2f} " + " ".join(
            f"{shares[m]:>8.1f}" for m in ("Top 1%", "Top 10%", "Top 20%", "Top 40%")
        ) + f" {wti:>6.2f}"
        print(line)

    best_beta, best_f, best_shares, best_wti = min(results, key=lambda r: r[1])

    print("-" * 78)
    print(f"  Bestes beta_bar (Raster): {best_beta:.4f}   f = {best_f:.2f}")
    print(f"  Anteile bei diesem beta vs. SOEP-Ziel:")
    print(f"    {'Moment':<9} {'Modell':>8} {'SOEP':>8} {'Diff':>8}  (Ziel=Zielmoment)")
    for m in ("Top 1%", "Top 10%", "Top 20%", "Top 40%"):
        tag = "Ziel" if m in TARGET_MOMENTS else "Diagnose"
        print(
            f"    {m:<9} {best_shares[m]:>8.1f} {target[m]:>8.1f} "
            f"{best_shares[m]-target[m]:>+8.1f}  {tag}"
        )
    print(
        f"  Vermoegen/Einkommen (Modell {best_wti:.2f} vs. SOEP "
        f"{SOEP_WEALTH_TO_INCOME[region]:.2f})"
    )
    print()
    return best_beta


def main() -> None:
    model = build_model()  # einmal bauen, fuer beide Regionen wiederverwenden

    # beta-Raster: unter 1/(1+r) bleiben (Rueckkehr-Ungeduld). Bei r=0.02 ist
    # 1/(1+r) ~ 0.980; wir bleiben knapp darunter.
    beta_grid = np.round(np.linspace(0.90, 0.976, 9), 4)

    estimated = {}
    for region in REGION_INPUTS:
        estimated[region] = estimate_region(model, region, beta_grid)

    print("=" * 78)
    print("Geschaetzte beta_bar:", {r: round(b, 4) for r, b in estimated.items()})
    print("=" * 78)
    print(
        "Lies das Profil: klares Minimum in f(beta) -> beta identifiziert.\n"
        "Flaches Profil -> beta durch die Anteile schwach identifiziert; dann ist\n"
        "das Vermoegen/Einkommen (V/E-Spalte) das bessere beta-Ziel."
    )


if __name__ == "__main__":
    main()
