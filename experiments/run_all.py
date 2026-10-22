"""Run implemented groups only; later groups remain explicitly skipped."""
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    runpy.run_path(str(PROJECT_ROOT / "experiments" / "01_detection" / "run_detection.py"), run_name="__main__")
    for group in ("02_overhead", "03_threshold", "04_ground_truth", "05_joint_decision", "06_long_term"):
        print(f"{group}: skipped - 该组实验尚未实现")


if __name__ == "__main__":
    main()
