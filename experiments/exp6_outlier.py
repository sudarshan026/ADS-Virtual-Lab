"""Experiment 6 — Outlier Detection."""
import sys, os

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base, "adsca exp 6.py")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, filepath, "exec"), {"__name__": "__experiment__", "__file__": filepath})
