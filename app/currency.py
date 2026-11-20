"""Currency conversion for international buyers. Gulf currencies are pegged to the US dollar (exact, never stale);
GBP/EUR/INR are fetched once at startup from the ECB (frankfurter.dev, no key) and skipped if that fails."""
import json
import logging
import urllib.request

PER_USD = {"USD": 1.0, "SAR": 3.75, "AED": 3.6725, "QAR": 3.64, "OMR": 0.3845, "BHD": 0.376}  # official pegs
RATES_DATE = "pegged"
try:
    with urllib.request.urlopen("https://api.frankfurter.dev/v1/latest?base=USD&symbols=GBP,EUR,INR", timeout=3) as r:
        _live = json.load(r)
    PER_USD |= _live["rates"]
    RATES_DATE = _live["date"]
except Exception as e:  # offline or API down: pegged currencies still work
    logging.getLogger("uvicorn.error").warning("live exchange rates unavailable (%s); GBP/EUR/INR disabled", e)


def to_sar(amount, cur):
    return round(amount * PER_USD["SAR"] / PER_USD[cur])


def from_sar(amount, cur):
    return amount * PER_USD[cur] / PER_USD["SAR"]


def fmt(amount):
    """1234567 -> '1.23M', 76500 -> '76.5K', 999600 -> '1M' (not '1e+03K')"""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if amount >= size * 0.9995:  # would round up to 1000 of the unit below
            return f"{float(f'{amount / size:.3g}'):g}{unit}"
    return f"{amount:.0f}"
