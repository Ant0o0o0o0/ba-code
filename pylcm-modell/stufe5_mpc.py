import numpy as np
import pandas as pd

from stufe2_schocks import (
    FIRST_AGE,
    INTEREST_RATE,
    REGION_INPUTS,
    build_model,
    make_params,
)

# Geschaetzte beta_bar aus Stufe 4 (bei Bedarf nach Verfeinerung aktualisieren).
BETA_EST = {"Ost": 0.9475, "West": 0.9570}

MPC_AGE = 45         # repraesentatives Erwerbsalter fuer die MPC-Momentaufnahme
WINDFALL = 0.10      # einmaliger Windfall ~10% eines Jahreseinkommens (~1)
N_AGENTS = 3000

GROUPS = ("Top 1%", "Top 10%", "Top 20%", "Top 40%", "untere 50%")


def simulate_panel(model, params, initial_df):
    result = model.simulate(
        params=params,
        initial_conditions=initial_df,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )
    return result.to_dataframe(additional_targets="all")


def mpc_by_top_groups(mpc, sortvar):
    """MPC fuer die reichsten Top 1/10/20/40% und die unteren 50% (nach sortvar).

    sortvar absteigend sortiert -> Index 0 ist der Reichste. "Top p%" = Mittel
    der ersten p% (kumulativ, wie bei Carroll); "untere 50%" = letzten 50%.
    """
    order = np.argsort(sortvar)[::-1]
    m = mpc[order]
    n = len(m)
    out = {}
    for p in (0.01, 0.10, 0.20, 0.40):
        k = max(1, int(round(p * n)))
        out[f"Top {int(p*100)}%"] = float(m[:k].mean())
    kb = max(1, int(round(0.50 * n)))
    out["untere 50%"] = float(m[n - kb:].mean())
    return out


def mpc_for_region(model, region):
    gamma = REGION_INPUTS[region]["gamma"]
    repl = REGION_INPUTS[region]["replacement_rate"]
    beta = BETA_EST[region]
    params = make_params(gamma=gamma, replacement_rate=repl, beta=beta)

    # --- Basislinie ab Alter 25 ---
    init25 = pd.DataFrame(
        {
            "regime_name": "working_life",
            "age": float(FIRST_AGE),
            "wealth": np.linspace(0.0, 2.0, N_AGENTS),
            "z_perm": 0.0,
            "tran_shock": 0.0,
        }
    )
    base = simulate_panel(model, params, init25)
    base["age"] = base["age"].astype(int)

    at_age = (
        base[base["age"] == MPC_AGE]
        .dropna(subset=["wealth", "consumption"])
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    # Windfall auf die MITTEL: Mittel = (1+r)*wealth + Einkommen -> wealth + x/(1+r).
    bump = WINDFALL / (1 + INTEREST_RATE)
    init_windfall = pd.DataFrame(
        {
            "regime_name": "working_life",
            "age": float(MPC_AGE),
            "wealth": at_age["wealth"].to_numpy() + bump,
            "z_perm": at_age["z_perm"].to_numpy(),
            "tran_shock": at_age["tran_shock"].to_numpy(),
        }
    )
    wf = simulate_panel(model, params, init_windfall)
    wf["age"] = wf["age"].astype(int)
    c_windfall = (
        wf[wf["age"] == MPC_AGE].sort_values("subject_id")["consumption"].to_numpy()
    )

    c_base = at_age["consumption"].to_numpy()
    mpc = (c_windfall - c_base) / WINDFALL

    return {
        "beta": beta,
        "overall": float(np.mean(mpc)),
        "by_wealth": mpc_by_top_groups(mpc, at_age["wealth"].to_numpy()),
        "by_income": mpc_by_top_groups(mpc, at_age["income"].to_numpy()),
    }


def main() -> None:
    model = build_model()

    print("=" * 66)
    print(f"MPC (Jahres-MPC) aus einmaligem Windfall x = {WINDFALL} bei Alter {MPC_AGE}")
    print("=" * 66)
    for region in REGION_INPUTS:
        r = mpc_for_region(model, region)
        print(
            f"\nRegion {region}   beta_bar = {r['beta']:.4f}   "
            f"MPC insgesamt = {r['overall']:.2f}"
        )
        print(f"  {'Gruppe':<12}{'nach Vermoegen':>16}{'nach Einkommen':>16}")
        print(f"  {'-'*12}{'-'*16}{'-'*16}")
        for g in GROUPS:
            print(
                f"  {g:<12}{r['by_wealth'][g]:>16.2f}{r['by_income'][g]:>16.2f}"
            )
    print()
    print("Erwartung (wie Carroll Tab. 3): MPC FAELLT mit dem Vermoegen --")
    print("Top 1% niedrig, untere 50% hoch. Gesamt-MPC im Bereich ~0.2-0.6.")


if __name__ == "__main__":
    main()
