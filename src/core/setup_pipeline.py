import os
import time
from utils.logger import get_logger
from utils.progress_tracker import progress_bar, status
from utils.config_loader import load_config
from utils.folder_creation import create_folder_structure, create_general_structure

def initialize_pipeline():
    """
    Initialize the Hypnos-6 pipeline:
    - Load configs
    - Create folder structures
    - Log environment info
    - Simulate first phase
    """

    # -----------------------------
    # Paths
    # -----------------------------
    CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(CURRENT_DIR, "config", "global_config.yaml")

    # -----------------------------
    # Logger Setup
    # -----------------------------
    logger = get_logger()
    logger.info("Initializing Hypnos-6 Environment...")

    # -----------------------------
    # Config Load
    # -----------------------------
    try:
        cfg = load_config(CONFIG_PATH)
        logger.info(f"Configuration loaded successfully from: {CONFIG_PATH}")
    except FileNotFoundError as e:
        logger.error(f"Configuration file missing: {e}")
        raise SystemExit("⚠️ Project configuration file not found. Exiting setup.")

    # -----------------------------
    # Folder Structure Setup
    # -----------------------------
    status("[cyan]Verifying folder structure...[/]")
    create_general_structure()
    create_folder_structure()
    status("[green]Folder structure verified![/]")
    logger.info("Folder structure verified and ready.")

    # -----------------------------
    # Phase 1 Simulation Demo
    # -----------------------------
    project_name = cfg.get("project_name", "AjiFinal_Hypnos6")
    device = cfg.get("device", "cpu")

    logger.info(f"🚀 Project: {project_name} initialized on device: {device}")
    status(f"[bold yellow]Starting Phase 1: Preprocessing Simulation...[/]")

    for i in progress_bar(range(10), "Running Phase 1 Preprocessing"):
        time.sleep(0.3)

    status("[green]✅ Phase 1 simulation completed successfully![/]")
    logger.info("Phase 1 preprocessing simulation completed successfully.")

    # -----------------------------
    # Closing
    # -----------------------------
    logger.info("🎯 Hypnos-6 Pipeline Initialization Finished Successfully.")
    status("[bold green]Project setup completed. Ready for Phase 1 implementation![/]")
