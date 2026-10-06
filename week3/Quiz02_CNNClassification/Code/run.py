from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.experiments import experiment_cli

if __name__ == "__main__":
    experiment_cli(2)
