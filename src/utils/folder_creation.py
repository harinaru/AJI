import os
from loguru import logger

def create_folder_structure(base_path: str = "results") -> list:
    phases = [
        "phase1_preprocessing",
        "phase2_feature_extraction",
        "phase3_nassi_core",
        "phase4_causal_analysis",
        "phase5_generative_anomaly",
        "phase6_xai",
        "phase7_decision_support",
    ]
    subfolders = ["logs", "figures", "tables", "models"]
    created_dirs = []
    for phase in phases:
        for sub in subfolders:
            path = os.path.join(base_path, phase, sub)
            os.makedirs(path, exist_ok=True)
            created_dirs.append(path)
    logger.info(f"Verified/created {len(created_dirs)} result directories under '{base_path}'")
    return created_dirs

def create_general_structure() -> None:
    folders = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/annotations",
        "config",
        "notebooks",
        "tests",
        "results",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    logger.info("Verified/created general project directory structure.")

def verify_all() -> None:
    create_general_structure()
    create_folder_structure()
    logger.success("✅ Project directory hierarchy verified successfully.")
