"""Experiment 5 — SMOTE Technique."""
import sys, os

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(base, "ADS_VirtualLab_SMOTE-main exp 5")
    # This experiment imports from utils/ and models/ inside its directory
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    filepath = os.path.join(exp_dir, "app.py")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, filepath, "exec"), {"__name__": "__experiment__", "__file__": filepath})
