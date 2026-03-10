"""
launch.py — Start MT5 if not running, wait until ready, then run live_trader.

Usage:
    python launch.py
"""

import subprocess
import sys
import time
import logging

import MetaTrader5 as mt5

from src.config import MT5_PATH, MT5_LOGIN

MT5_EXE = MT5_PATH
RETRY_INTERVAL = 5   # seconds between connection attempts
MAX_WAIT = 240        # seconds to wait for MT5 to become ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def is_mt5_running() -> bool:
    """Check if terminal64.exe is already running."""
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
        capture_output=True, text=True
    )
    return "terminal64.exe" in result.stdout


def wait_for_mt5_ready(max_wait: int = MAX_WAIT) -> bool:
    """
    Try to initialise MT5 and get account info repeatedly until success
    or max_wait seconds have elapsed.
    """
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        if mt5.initialize(path=MT5_EXE):
            info = mt5.account_info()
            if info is not None and info.login == MT5_LOGIN:
                log.info(
                    f"MT5 ready — login={info.login} | server={info.server} | "
                    f"balance={info.balance:.2f} {info.currency}"
                )
                mt5.shutdown()   # live_trader will re-connect on its own
                return True
            mt5.shutdown()
        log.info(f"Waiting for MT5... (attempt {attempt}, {int(deadline - time.time())}s left)")
        time.sleep(RETRY_INTERVAL)

    log.error(f"MT5 did not become ready within {max_wait}s.")
    return False


def main():
    if is_mt5_running():
        log.info("MT5 is already running.")
    else:
        log.info(f"MT5 not running — launching {MT5_EXE}")
        subprocess.Popen([MT5_EXE])

    if not wait_for_mt5_ready():
        sys.exit(1)

    log.info("Starting live_trader...")
    result = subprocess.run(
        [sys.executable, "-m", "src.execution.live_trader"],
        cwd=r"C:\Users\Administrator\IdeaProjects\HMM_XGBoost_H1_Swing",
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
