"""Experiment 7 — Time Series Forecasting."""
import sys, os

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(base, "ADS_Virtual_Lab-main exp 7")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    # Change working dir so CSV data files load correctly
    prev_cwd = os.getcwd()
    os.chdir(exp_dir)
    try:
        filepath = os.path.join(exp_dir, "exp7.py")
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, filepath, "exec"), {"__name__": "__experiment__", "__file__": filepath})
    finally:
        os.chdir(prev_cwd)
