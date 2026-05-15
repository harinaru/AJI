from loguru import logger

import os
import sys
import json
import platform
import psutil
import datetime

from typing import Any, Dict, Optional


# ============================================================
# CONFIGURATION
# ============================================================

LOG_DIR = os.path.join(
    "results",
    "logs",
)

os.makedirs(
    LOG_DIR,
    exist_ok=True,
)

timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

log_path = os.path.join(
    LOG_DIR,
    f"run_{timestamp}.log",
)


# ============================================================
# REMOVE DEFAULT LOGGER
# ============================================================

logger.remove()


# ============================================================
# CONSOLE LOGGER
# ============================================================

logger.add(

    sys.stdout,

    level="INFO",

    colorize=True,

    enqueue=True,

    backtrace=True,

    diagnose=True,

    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level:<8} | "
        "{message}"
    ),
)


# ============================================================
# FILE LOGGER
# ============================================================

logger.add(

    log_path,

    rotation="20 MB",

    retention="30 days",

    level="DEBUG",

    encoding="utf-8",

    enqueue=True,

    backtrace=True,

    diagnose=True,

    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level:<8} | "
        "{module}:{function}:{line} - "
        "{message}"
    ),
)


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def _get_conda_env() -> Optional[str]:

    return (
        os.environ.get("CONDA_DEFAULT_ENV")
        or
        os.environ.get("VIRTUAL_ENV")
    )


def _get_cpu_info() -> Dict[str, Any]:

    try:

        cpu_count = psutil.cpu_count(
            logical=True
        )

        cpu_freq = psutil.cpu_freq()

        mem = psutil.virtual_memory()

        return {

            "cpu_cores":
                cpu_count,

            "cpu_frequency_MHz":
                (
                    round(cpu_freq.current, 2)
                    if cpu_freq
                    else None
                ),

            "memory_total_GB":
                round(
                    mem.total / (1024 ** 3),
                    2,
                ),
        }

    except Exception as e:

        return {

            "error":
                f"CPU info unavailable: {e}"
        }


def _get_installed_packages_summary(
    limit: int = 25,
) -> Dict[str, Any]:

    try:

        import pkg_resources

        packages = sorted([
            (
                d.project_name,
                d.version,
            )
            for d
            in pkg_resources.working_set
        ])

        return {

            "total_packages":
                len(packages),

            "top_packages":
                packages[:limit],
        }

    except Exception as e:

        return {

            "error":
                f"Package list unavailable: {e}"
        }


def _get_python_env_summary() -> Dict[str, Any]:

    info: Dict[str, Any] = {

        "timestamp":
            timestamp,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "architecture":
            platform.architecture()[0],

        "processor":
            platform.processor(),

        "conda_env":
            _get_conda_env(),

        "working_directory":
            os.getcwd(),

        "executable":
            sys.executable,

        "reproducibility_seed":
            42,
    }

    info.update(
        _get_cpu_info()
    )

    info.update(
        _get_installed_packages_summary(
            limit=30
        )
    )

    return info


def _write_env_snapshot() -> str:

    info_path = os.path.join(

        LOG_DIR,

        f"env_snapshot_{timestamp}.json",
    )

    try:

        env_info = (
            _get_python_env_summary()
        )

        with open(

            info_path,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                env_info,

                f,

                indent=2,
            )

        logger.debug(
            f"Environment snapshot written to {info_path}"
        )

    except Exception as e:

        logger.warning(
            f"Failed to write environment snapshot: {e}"
        )

    return info_path


# ============================================================
# LOGGER ENTRY POINT
# ============================================================

def get_logger() -> Any:

    logger.info("=" * 80)
    logger.info(
        "Initializing AjiFinal Logging System"
    )
    logger.info("=" * 80)

    snapshot_path = (
        _write_env_snapshot()
    )

    env_info = (
        _get_python_env_summary()
    )

    logger.info(

        f"Platform: {env_info['platform']} | "
        f"CPU: {env_info['processor']} | "
        f"RAM: {env_info['memory_total_GB']} GB"
    )

    logger.info(

        f"Python: "
        f"{env_info['python_version'].split()[0]} | "
        f"Conda Env: "
        f"{env_info['conda_env']}"
    )

    logger.info(

        f"Packages: "
        f"{env_info['total_packages']} total"
    )

    logger.info(
        f"Environment snapshot: {snapshot_path}"
    )

    logger.info(
        f"Working Directory: "
        f"{env_info['working_directory']}"
    )

    logger.info(
        f"Executable: "
        f"{env_info['executable']}"
    )

    logger.info(
        f"Reproducibility Seed: "
        f"{env_info['reproducibility_seed']}"
    )

    logger.info(
        f"Log File Path: {log_path}"
    )

    logger.success(
        "[DONE] Logging initialized successfully."
    )

    logger.info("=" * 80)

    return logger