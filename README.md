# Code zur Bachelorarbeit — Regionale Heterogenität in der marginalen Konsumquote auf Basis der Vermögensheterogenität - von Anton Uni Bonn 2026
>
> 1. Die Modell-Skripte im Ordner [`pylcm-modell/`](pylcm-modell/) laufen im
>    Repository **PyLCM** (<https://github.com/OpenSourceEconomics/pylcm>).
> 2. Die Datenaufbereitungs-Skripte im Ordner
>    [`soep-aufbereitung/`](soep-aufbereitung/) laufen im Repository
>    **soep-preparation** (https://github.com/ttsim-dev/soep-preparation/?tab=readme-ov-file).
>
> Um den Code auszuführen, werden die Skripte in eine lokale Kopie des jeweiligen
> Repositories gelegt (siehe unten). Beide Repositories verwenden
> [pixi](https://pixi.sh/) zur Verwaltung der Softwareumgebung.

---

## Ablauf im Überblick

Der Datenfluss verläuft von der Aufbereitung zur Modellierung:

```
SOEP-Rohdaten (V41, lizenziert, NICHT enthalten)
        │
        ▼
soep-aufbereitung/   →   Zielmomente (Vermögensanteile),
                         Alters-Einkommensprofil (γ), Ersatzrate
        │
        ▼   (Zahlen werden in die Modell-Skripte übertragen)
        │
pylcm-modell/        →   Schätzung von β̄  →  MPC
```

Die aus der Datenaufbereitung gewonnenen Kennzahlen (Vermögensanteile pro Region,
Einkommenswachstum γ, Ersatzrate) sind in den Modell-Skripten als Zahlenwerte
hinterlegt. Die beiden Teile sind dadurch **entkoppelt**: Die Modellierung läuft
ohne Zugriff auf die SOEP-Rohdaten.

---

## Teil 1 — Datenaufbereitung (Ordner `soep-aufbereitung/`)

Diese Skripte gehören in eine lokale Kopie des **soep-preparation**-Repositories.

**Ausführen** (in der lokalen soep-preparation-Kopie):

```console
$ pixi run pytask     # baut die Pipeline (führt die task_-Skripte aus)
```

Die `analyse_`-Skripte müssen anschließend ausgeführt werden bzw. sind in die Pipeline
eingebunden.

> **SOEP-Rohdaten sind nicht enthalten.** Die Skripte benötigen die lizenzierten
> SOEP-Core-Rohdaten (Version 41) im Verzeichnis `soep_preparation/data/V41`.
> Diese Daten dürfen aus lizenzrechtlichen Gründen **nicht** weitergegeben werden
> und sind daher nicht Teil dieses Repositories.

---

## Teil 2 — Modell (Ordner `pylcm-modell/`)

Diese Skripte gehören in eine lokale Kopie des **PyLCM**-Repositories und werden
dort im Projektstammverzeichnis abgelegt.

| Datei | Inhalt |
|---|---|
| `stufe2_schocks.py` | **Modelldefinition**: Regime (Erwerbsleben, Ruhestand, Ende), Nutzen-, Einkommens- und Budgetfunktionen, Kalibrierung. Erzeugt beim Ausführen die Abbildung `stufe2c_vermoegen.html` (Vermögensprofil über das Alter). Wird von den beiden folgenden Skripten importiert. |
| `stufe4_schaetzung.py` | **Schätzung** des Diskontfaktors β̄ pro Region durch Anpassung an die Vermögensanteile (Rastersuche, gibt das β-Profil aus). |
| `stufe5_mpc.py` | **MPC**: Windfall-Experiment beim geschätzten β̄; berechnet die aggregierte MPC und die Aufschlüsselung nach Vermögens- und Einkommensgruppen. |
| `stufe2c_vermoegen.html` | Beispiel-Ausgabe von `stufe2_schocks.py` (kann neu erzeugt werden). |

**Ausführen** (in der lokalen PyLCM-Kopie, Skripte im Stammverzeichnis):

```console
$ pixi run python stufe2_schocks.py      # Modell + Vermögensprofil
$ pixi run python stufe4_schaetzung.py   # β̄ schätzen
$ pixi run python stufe5_mpc.py          # MPC berechnen
```

**Hinweis zu den Importen.** `stufe4_schaetzung.py` und `stufe5_mpc.py`
importieren das Modell aus `stufe2_schocks.py`.

 ## Software

Beide Teile nutzen `pixi` zur Umgebungsverwaltung; die genauen Abhängigkeiten
liegen in der `pixi.toml`/`pixi.lock` des jeweiligen übergeordneten Repositories
(PyLCM bzw. soep-preparation). PyLCM baut auf `jax` auf und benötigt Python ≥ 3.14.

## Datengrundlage

Sozio-oekonomisches Panel (SOEP-Core, Version 41), DIW Berlin.
