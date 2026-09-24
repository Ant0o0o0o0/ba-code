import numpy as np
import pandas as pd
import jax.numpy as jnp
import plotly.express as px
from pprint import pprint

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


# ---------------------------------------------------------------------------
# 1) Regime
# ---------------------------------------------------------------------------
@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


FIRST_AGE = 25
RETIREMENT_AGE = 67  # erstes Rentenjahr; letztes Erwerbsjahr 66
FINAL_AGE = 80


# ---------------------------------------------------------------------------
# 2) Modellfunktionen
# ---------------------------------------------------------------------------
def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def income_working(
    age: float,
    z_perm: ContinuousState,
    tran_shock: ContinuousState,
    gamma: float,
) -> FloatND:
    """Einkommensniveau = deterministisches Profil * permanent * transitorisch."""
    return (
        jnp.power(gamma, age - FIRST_AGE)
        * jnp.exp(z_perm)
        * jnp.exp(tran_shock)
    )


def pension(replacement_rate: float, gamma: float, retirement_age: float) -> FloatND:
    """Flache Rente = Ersatzrate * deterministisches Einkommen bei Renteneintritt."""
    return replacement_rate * jnp.power(gamma, (retirement_age - 1) - FIRST_AGE)


def resources_working(
    wealth: ContinuousState, interest_rate: float, income: FloatND
) -> FloatND:
    """Verfuegbare Mittel in DIESER Periode."""
    return (1 + interest_rate) * wealth + income


def resources_retirement(
    wealth: ContinuousState, interest_rate: float, pension: FloatND
) -> FloatND:
    return (1 + interest_rate) * wealth + pension


def next_wealth_working(
    resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    return resources - consumption


def next_wealth_retirement(
    resources: FloatND, consumption: ContinuousAction
) -> ContinuousState:
    return resources - consumption


def borrowing_constraint(
    resources: FloatND, consumption: ContinuousAction
) -> BoolND:
    """Kein Verschulden: Konsum hoechstens die verfuegbaren Mittel."""
    return consumption <= resources


def next_regime_from_working(age: float, retirement_age: float) -> ScalarInt:
    return jnp.where(
        age >= retirement_age - 1, RegimeId.retirement, RegimeId.working_life
    )


def next_regime_from_retirement(age: float, final_age: float) -> ScalarInt:
    return jnp.where(age >= final_age - 1, RegimeId.dead, RegimeId.retirement)


# ---------------------------------------------------------------------------
# 3) Modell
# ---------------------------------------------------------------------------
N_Z_POINTS = 7      # Stuetzstellen persistentes Einkommen
N_TRAN_POINTS = 5   # Stuetzstellen transitorischer Schock
# Hinweis: Zustandsraum = wealth x z_perm x tran_shock. Punkte sparsam halten,
# sonst wird Stufe 4 (viele Loesungen in der Schleife) langsam.


def build_model() -> Model:
    ages = AgeGrid(start=FIRST_AGE, stop=FINAL_AGE, step="Y")
    wealth_grid = LinSpacedGrid(start=0.0, stop=30.0, n_points=40)
    consumption_grid = LogSpacedGrid(start=0.05, stop=30.0, n_points=100)

    working_life = Regime(
        transition=next_regime_from_working,
        active=lambda age: age < RETIREMENT_AGE,
        states={
            "wealth": wealth_grid,
            "z_perm": RouwenhorstAR1Process(n_points=N_Z_POINTS),
            "tran_shock": NormalIIDProcess(
                n_points=N_TRAN_POINTS, gauss_hermite=True
            ),
        },
        state_transitions={"wealth": next_wealth_working},
        actions={"consumption": consumption_grid},
        functions={
            "utility": utility,
            "income": income_working,
            "resources": resources_working,
        },
        constraints={"borrowing_constraint": borrowing_constraint},
    )

    retirement = Regime(
        transition=next_regime_from_retirement,
        active=lambda age: RETIREMENT_AGE <= age < FINAL_AGE,
        states={"wealth": wealth_grid},
        state_transitions={"wealth": next_wealth_retirement},
        actions={"consumption": consumption_grid},
        functions={
            "utility": utility,
            "pension": pension,
            "resources": resources_retirement,
        },
        constraints={"borrowing_constraint": borrowing_constraint},
    )

    dead = Regime(
        transition=None,
        active=lambda age: age >= FINAL_AGE,
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        description="Stufe 2c: persistentes + transitorisches Einkommen, Assets-Timing.",
    )


# ---------------------------------------------------------------------------
# 4) Parameter
# ---------------------------------------------------------------------------
BETA = 0.96                       # Platzhalter -- geschaetzt in Stufe 4
INTEREST_RATE = 0.02              # kleiner positiver realer Zins

RHO = 0.98                        # Persistenz des log-Einkommens
SIGMA_PERM = float(np.sqrt(0.016))  # ~0.1265  (deine Varianz 0.016)
VAR_TRAN = 0.05                   # deine Varianz des transitorischen Schocks
SIGMA_TRAN = float(np.sqrt(VAR_TRAN))  # ~0.2236
MU_TRAN = -VAR_TRAN / 2           # mittelwert-1-lognormal: mu = -sigma^2/2

REGION_INPUTS = {
    "Ost":  {"gamma": 1.0035, "replacement_rate": 0.978},
    "West": {"gamma": 1.0042, "replacement_rate": 0.899},
}


def make_params(gamma: float, replacement_rate: float, beta: float = BETA) -> dict:
    """beta ist Argument, damit Stufe 4 es in der Schaetzschleife variieren kann."""
    return {
        "discount_factor": beta,
        "interest_rate": INTEREST_RATE,
        "working_life": {
            "income": {"gamma": gamma},
            "next_regime": {"retirement_age": float(RETIREMENT_AGE)},
            "z_perm": {"rho": RHO, "sigma": SIGMA_PERM, "mu": 0.0},
            "tran_shock": {"mu": MU_TRAN, "sigma": SIGMA_TRAN},
        },
        "retirement": {
            "pension": {
                "replacement_rate": replacement_rate,
                "gamma": gamma,
                "retirement_age": float(RETIREMENT_AGE),
            },
            "next_regime": {"final_age": float(FINAL_AGE)},
        },
    }


# ---------------------------------------------------------------------------
# 5) Loesen, simulieren, pruefen
# ---------------------------------------------------------------------------
def wealth_shares(wealth: np.ndarray, percents=(0.01, 0.10, 0.20, 0.40)) -> dict:
    """Anteil des Gesamtvermoegens, den die reichsten p% halten (in Prozent)."""
    w = np.sort(np.maximum(wealth, 0.0))[::-1]
    tot = w.sum()
    return {
        f"Top {int(p*100)}%": round(100 * w[: max(1, int(p * len(w)))].sum() / tot, 1)
        for p in percents
    }


def main() -> None:
    model = build_model()

    print("=" * 70)
    print("PARAMETER-VORLAGE (falls make_params abweicht: hiernach richten):")
    print("=" * 70)
    pprint(dict(model.get_params_template()))
    print()

    n_agents = 3000
    frames = []
    for region, inp in REGION_INPUTS.items():
        params = make_params(**inp)

        initial_df = pd.DataFrame(
            {
                "regime_name": "working_life",
                "age": float(FIRST_AGE),
                "wealth": np.linspace(0.0, 2.0, n_agents),
                "z_perm": 0.0,
                "tran_shock": 0.0,
            }
        )

        result = model.simulate(
            params=params,
            initial_conditions=initial_df,
            period_to_regime_to_V_arr=None,
            log_level="debug",
            seed=0,
        )

        df = result.to_dataframe(additional_targets="all")
        df["age"] = df["age"].astype(int)
        df["region"] = region
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    # Terminales "dead"-Regime hat kein wealth -> NaN-Zeilen entfernen.
    data = data.dropna(subset=["wealth", "consumption"])

    # 5a) Vermoegensprofil ueber das Alter
    mean_by_age = (
        data.groupby(["region", "age"], as_index=False)[["wealth", "consumption"]]
        .mean(numeric_only=True)
    )
    fig = px.line(
        mean_by_age, x="age", y="wealth", color="region",
        title="Durchschnittliches Nettovermoegen ueber das Alter", markers=True,
    )
    fig.write_html("stufe2c_vermoegen.html")

    # 5b) Vermoegensanteile vs. deine SOEP-Ziele
    soep = {
        "Ost":  {"Top 1%": 17.61, "Top 10%": 52.61, "Top 20%": 73.54, "Top 40%": 95.15},
        "West": {"Top 1%": 17.51, "Top 10%": 52.15, "Top 20%": 70.87, "Top 40%": 92.17},
    }
    print("=" * 70)
    print("Vermoegensanteile: Modell vs. SOEP-Ziel (Prozent des Gesamtvermoegens)")
    print("=" * 70)
    for region in REGION_INPUTS:
        w = data.loc[data["region"] == region, "wealth"].to_numpy()
        model_shares = wealth_shares(w)
        print(f"\n  {region}:")
        print(f"    {'Moment':<10} {'Modell':>8} {'SOEP':>8} {'Diff':>8}")
        for k in ("Top 1%", "Top 10%", "Top 20%", "Top 40%"):
            m, s = model_shares[k], soep[region][k]
            print(f"    {k:<10} {m:>8.1f} {s:>8.1f} {m-s:>+8.1f}")

    print()
    print("Grafik gespeichert: stufe2c_vermoegen.html")


if __name__ == "__main__":
    main()