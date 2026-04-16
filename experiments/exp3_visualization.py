"""Experiment 3 — Data Visualization."""
import sys, os

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base, "app exp 3.py")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    ns = {"__name__": "__experiment__", "__file__": filepath}
    exec(compile(code, filepath, "exec"), ns)
    # The original has a main() function guarded by if __name__ == "__main__"
    # which won't fire because __name__ is "__experiment__".
    # Call main() explicitly.
    if "main" in ns:
        ns["main"]()
