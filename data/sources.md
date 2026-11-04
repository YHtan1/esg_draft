# Dated real-data initialization test

Snapshot: **31 October 2025**. Data are cached; notebook execution needs no network.
This is a real-data initialized risk-neutral simulation test, not a full market
calibration or physical-measure historical forecast/backtest.

| Input | Source | Treatment |
| --- | --- | --- |
| Thai zero curve | [ThaiBMA dated data](https://www.thaibma.or.th/yieldcurve/zero/2025-10-31) | 54 published zero nodes; annual-effective conversion cross-checked from simultaneous short-tenor BEY observations |
| Thai bond yield observations | Same ThaiBMA response, `Stat` | Individual dealer quotations, displayed separately from zero rates; publisher-flagged synthetic quotations excluded from the raw-quote plot |
| US zero curve | [Federal Reserve GSW](https://www.federalreserve.gov/data/nominal-yield-curve.htm) | 30 continuously compounded SVENY01–SVENY30 nodes from the dated CSV row |
| US par observations | [US Treasury feed](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed) | Dated par-yield points, displayed as par observations rather than zero rates |
| SET spot | [SET October 2025 report](https://www.set.or.th/en/market/news-and-alert/newsdetails?id=99380000&symbol=SET) | 1,309.50; report published 5 November 2025 describing the October month end |
| S&P 500 spot | [Yahoo Finance daily observations](https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=1759276800&period2=1761955200&interval=1d) | 6,840.20 month-end spot from 23 daily closes in October, cross-checked against [FRED](https://fred.stlouisfed.org/data/SP500) and an [issuer SEC filing](https://www.sec.gov/Archives/edgar/data/895421/000183988225063495/ms11118_424b2-36644.htm) |

ThaiBMA's zero nodes are already constructed from government quotations; the
Federal Reserve GSW nodes are fitted research estimates. Neither is a set of
individually traded zero-coupon bond prices. Raw Thai dealer YTM and Treasury par
rates are different quantities and are not treated as zero rates.

Thai compounding is an explicit inference checked numerically: the first four
published zero nodes agree with `100*((1+BEY_pct/200)**2-1)` to within 0.000002
percentage points. We use `P(0,T)=(1+zero_pct/100)**(-T)`. The US CSV header
explicitly specifies continuous compounding, so `P(0,T)=exp(-zero_pct/100*T)`.
The notebook displays the convention checks. Its interpolation is piecewise
linear in log discounts, not an attempt to reproduce the publisher's interpolation.
The US zero rate below one year is assumed flat; the last instantaneous forward
is held flat beyond each curve's final knot.

Heston variance/mean reversion/vol-of-vol/correlation, rate volatilities and mean
reversions, FX spot/volatility, and the full Brownian correlation matrix remain
declared research assumptions. SET's observed 3.76% trailing dividend ratio is
carried as a constant forward model assumption. No option surface or SET daily
history was loaded. The S&P month of observations is not used to claim Heston
parameter identification. Corporate securities are representative coupon bonds
with assumed intensity/CIR/recovery, not calibrated named issues.

The JSON files beside this document are the complete offline inputs used by the
notebook. `thai_curve.json` and `us_curve.json` contain the original source URLs,
compounding conventions and hashes of the retrieved source files. The hashes
identify the retrieved vintage, rather than a live feed. `us_par_yields.json` is
shown separately from zero rates. `stock_indices.json` contains observed spots.
`snapshot.json` describes the vintage; `assumptions.json` contains the model inputs
that are not calibrated to observations. Edit those files to change the experiment.

The former raw-download cache and refresh scripts were backed up outside this
project during simplification. Notebook execution never fetches or silently refreshes data.
