"""Experiment 1 — Descriptive & Inferential Statistics."""
import sys, os

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(base, "ADS-exp-1-virtual-lab-main")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    filepath = os.path.join(exp_dir, "app.py")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, filepath, "exec"), {"__name__": "__experiment__", "__file__": filepath})
