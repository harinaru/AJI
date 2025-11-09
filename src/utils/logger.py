from loguru import logger
import os
import sys
import json
import platform
import pkgutil
import psutil
import datetime
import subprocess
from typing import Any, Dict, List, Optional

# ============================================================
# Configuration
# ============================================================

LOG_DIR = os.path.join("results", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"run_{timestamp}.log")

# ============================================================
# Loguru setup
# ============================================================

logger.remove()

# Console output
logger.add(lambda msg: print(msg, end=""), level="INFO")

# File output
logger.add(
    log_path,
    rotation="20 MB",
    retention="30 days",
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)

# ============================================================
# Environment snapshot utility
# ============================================================

def _get_conda_env() -> Optional[str]:
    """Get current conda environment name, if available."""
    return os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")


def _get_cpu_info() -> Dict[str, Any]:
    """Fetch CPU and system info using psutil."""
    try:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        return {
            "cpu_cores": cpu_count,
            "cpu_frequency_MHz": cpu_freq.current if cpu_freq else None,
            "memory_total_GB": round(mem.total / (1024**3), 2)
        }
    except Exception as e:
        return {"error": f"CPU info unavailable: {e}"}


def _get_installed_packages_summary(limit: int = 25) -> Dict[str, Any]:
    """Collect a brief summary of installed packages."""
    try:
        import pkg_resources
        packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
        return {
            "total_packages": len(packages),
            "top_packages": packages[:limit]
        }
    except Exception as e:
        return {"error": f"Package list unavailable: {e}"}


def _get_python_env_summary() -> Dict[str, Any]:
    """Collect metadata about the runtime environment."""
    info: Dict[str, Any] = {
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "conda_env": _get_conda_env(),
        "working_directory": os.getcwd(),
        "executable": sys.executable,
        "reproducibility_seed": 42,  # standard seed
    }
    info.update(_get_cpu_info())
    info.update(_get_installed_packages_summary(limit=30))
    return info


def _write_env_snapshot() -> str:
    """Write a detailed JSON environment snapshot."""
    info_path = os.path.join(LOG_DIR, f"env_snapshot_{timestamp}.json")
    try:
        env_info = _get_python_env_summary()
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)
        logger.debug(f"Environment snapshot written to {info_path}")
    except Exception as e:
        logger.warning(f"Failed to write environment snapshot: {e}")
    return info_path


# ============================================================
# Logger entry point
# ============================================================

from loguru import logger
from typing import Any

def get_logger() -> Any:
    logger.info("=" * 80)
    logger.info("🔍 Initializing AjiFinal Logging System")
    logger.info("=" * 80)

    snapshot_path = _write_env_snapshot()
    env_info = _get_python_env_summary()

    logger.info(f"Platform: {env_info['platform']} | CPU: {env_info['processor']} | RAM: {env_info['memory_total_GB']} GB")
    logger.info(f"Python: {env_info['python_version'].split()[0]} | Conda Env: {env_info['conda_env']}")
    logger.info(f"Packages: {env_info['total_packages']} total | Snapshot saved → {snapshot_path}")
    logger.info(f"Working Dir: {env_info['working_directory']}")
    logger.info(f"Executable: {env_info['executable']}")
    logger.info(f"Reproducibility Seed: {env_info['reproducibility_seed']}")

    logger.success("✅ Logging initialized successfully.\n")
    return logger
