# Six-security scenario notebook

Open **[six_securities.ipynb](six_securities.ipynb)** in VS Code. Select **Six Securities (.venv)** (or `.venv/bin/python`), then **Restart Kernel → Run All**. The notebook is the only workflow and contains all visual checks. Its saved charts are also available in [outputs/six_securities.html](outputs/six_securities.html).

The default projection is **1,000 scenarios × 120 monthly steps over 10 years**, with 121 reporting dates including today. It uses a dated **31 October 2025** snapshot of Thai/US curves and stock-index levels. Dynamics, FX and corporate-credit inputs are explicitly assumed; these are THB risk-neutral scenarios, not investment forecasts.

| Thai securities | US securities, converted to THB |
| --- | --- |
| Equity: SET proxy | Equity: S&P 500 proxy |
| Government: 10-year zero-coupon bond | Government: 10-year zero-coupon bond |
| Corporate: representative 10-year coupon bond | Corporate: representative 10-year coupon bond |

That is **six securities**. FX is a conversion state, not another security. The corporate bonds pay 4.5% annual coupons and assume 40% recovery; they are illustrative issues. Security dividends, coupons, principal and recovery are included in the martingale calculation. There are no insurance liabilities.

## Where to look

```text
six_securities.ipynb   Start here: settings, simulations, graphs and visual checks
models/               Financial model formulas and the six-security simulator
  market.py           Load data/ and validate model inputs, drivers and correlation
  simulation.py       Joint simulation and security distribution accounting
  curves.py           Discount curves and their interpolation
  rates.py            Hull–White / G2++ rates and Gaussian rate integrals
  equity.py           Heston methods and analytic QuantLib reference prices
  credit.py           Corporate valuation, intensity, coupons and recovery
  _random.py          Shared internal random-shock addressing
  __init__.py         Package marker
data/                Small offline JSON inputs and sources.md
outputs/             Generated charts, CSV tables, parameters and HTML reading copy
requirements.txt      One pinned Python/notebook environment
```

The notebook runs in this order:

1. **Observed data:** curve nodes, interpolation and separate dealer/par-yield points.
2. **Six-security projection:** monthly paths; optional controls for rate model, Heston scheme, scenario count and seed.
3. **Martingale check:** normalized discounted cumulative gains, target 1, sampling intervals and a terminal comparison.
4. **Heston comparison:** full truncation, reflection, QE and QE-martingale; constant-rate option prices versus QuantLib.
5. **Mean convergence:** independent-run errors compared with the `1/sqrt(N)` guide. This separate experiment uses sample sizes up to 4,000; the main projection remains 1,000.
6. **Timestep comparison:** monthly, twice-monthly and weekly internal steps.
7. **Export:** readable tables, parameters, source information and figures in `outputs/`.

All plot functions, experiments and diagnostic tables are in the notebook beside their results. `models/` has no separate test or benchmark workflow. QE methods are standalone Heston comparisons; the correlated six-security simulator supports full truncation and reflection. Button clicks update the projection and martingale plots; rerun later sections to refresh their results after changing controls.

## Environment

The local `.venv` is already installed. If recreating it, use the VS Code terminal from this folder (verified with Python 3.9):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name six-securities --display-name "Six Securities (.venv)"
```

Install the VS Code **Python** and **Jupyter** extensions. There is no editable-package installation or compiler step. Run All takes longer than just the main projection because the convergence and timestep sections repeat simulations. The ordinary saved plots remain readable without widgets.

See [data/sources.md](data/sources.md) for provenance and assumptions. Notebook execution works offline. The input data are dated snapshots, not live feeds.
