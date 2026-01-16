"""
PHASE 7 — MAIN CONTROLLER
CAP-Gated Temporal Normality Model

This controller:
✔ Enforces strict experiment order
✔ Prevents cherry-picked execution
✔ Is CPU/GPU agnostic
✔ Is Pyright / Pylance clean
✔ Guarantees reproducibility
"""

from .config import Config

# Import modules (NOT symbols) → Pyright-safe
from . import train as train_module
from . import infer as infer_module
from . import visualize as visualize_module


def main() -> None:
    print("\n" + "=" * 80)
    print(" PHASE 7 — FULL EXPERIMENT PIPELINE ")
    print("=" * 80)

    # -------------------------------------------------
    # LOAD CONFIGURATION (single source of truth)
    # -------------------------------------------------
    config = Config()
    config.create_directories()
    config.print_experiment_summary()

    # -------------------------------------------------
    # STEP 1: TRAINING (CONTROLS ONLY)
    # -------------------------------------------------
    print("\n▶ STEP 1: TRAINING MODELS")
    train_module.main()

    # -------------------------------------------------
    # STEP 2: INFERENCE (FROZEN WEIGHTS)
    # -------------------------------------------------
    print("\n▶ STEP 2: RUNNING INFERENCE")
    infer_module.main()

    # -------------------------------------------------
    # STEP 3: VISUALIZATION (RESEARCH-GRADE)
    # -------------------------------------------------
    print("\n▶ STEP 3: GENERATING FIGURES")
    visualize_module.main()

    print("\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
