#!/usr/bin/env python3
"""Prepare Ollama + a Qwen model for the LLM advisor (OpenAI-compatible API at :11434/v1).

What this does
---------------
1. Locates the ``ollama`` CLI (PATH / ``ollama.exe``, or default Windows install under ``%LOCALAPPDATA%``).
2. If nothing answers on http://127.0.0.1:11434, starts ``ollama serve`` in the background.
3. Runs ``ollama pull <model>`` so weights are present (can take a while).
4. Prints the ``improvement`` YAML snippet to use in ``config/default.yaml``.

Usage
-----
  .venv\\Scripts\\python scripts\\start_ollama_qwen.py
  .venv\\Scripts\\python scripts\\start_ollama_qwen.py --model qwen2.5:14b
  .venv\\Scripts\\python scripts\\start_ollama_qwen.py --skip-pull
  .venv\\Scripts\\python scripts\\start_ollama_qwen.py --skip-serve
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx in this environment: pip install httpx", file=sys.stderr)
    sys.exit(1)

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "qwen2.5:7b"
HEALTH_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version"
OPENAI_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1"


def _find_ollama_exe() -> str | None:
    """Resolve ``ollama`` CLI. IDE terminals often omit user PATH; Windows has a standard install dir."""
    for name in ("ollama", "ollama.exe"):
        found = shutil.which(name)
        if found:
            return found
    if sys.platform != "win32":
        return None
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates.append(Path(local) / "Programs" / "Ollama" / "ollama.exe")
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates.append(Path(pf) / "Ollama" / "ollama.exe")
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _ollama_ok() -> bool:
    try:
        r = httpx.get(HEALTH_URL, timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _start_ollama_serve_background() -> None:
    exe = _find_ollama_exe()
    if not exe:
        raise FileNotFoundError("ollama executable not found on PATH")
    kwargs: dict = {
        "args": [exe, "serve"],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        # Detached child so closing this terminal does not kill the server (best-effort).
        CREATE_NO_WINDOW = 0x08000000
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        if hasattr(subprocess, "DETACHED_PROCESS"):
            flags |= subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(**kwargs)  # noqa: S603


def _wait_for_server(timeout_s: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _ollama_ok():
            return True
        time.sleep(0.5)
    return False


def _run_ollama_pull(model: str) -> int:
    exe = _find_ollama_exe()
    if not exe:
        return 127
    return subprocess.call([exe, "pull", model])  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Ollama (if needed) and pull a Qwen model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag to pull (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--skip-serve",
        action="store_true",
        help="Do not try to start ``ollama serve`` if the API is down.",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip ``ollama pull`` (only check/start server).",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=45.0,
        help="Seconds to wait for the API after starting serve.",
    )
    args = parser.parse_args()

    ollama_exe = _find_ollama_exe()
    if not ollama_exe:
        print(
            "Could not find the ``ollama`` executable.\n"
            "  • Install: https://ollama.com/download\n"
            "  • Or add Ollama to your user PATH (default install:\n"
            "    %LOCALAPPDATA%\\Programs\\Ollama)\n"
            "  • IDE terminals sometimes omit PATH — try an external PowerShell window.",
            file=sys.stderr,
        )
        return 1

    if _ollama_ok():
        print(f"Ollama API is already up ({HEALTH_URL}).")
    elif args.skip_serve:
        print(
            f"Ollama API is not responding at {HEALTH_URL}. "
            "Start the Ollama app (Windows tray) or run ``ollama serve``, then retry.",
            file=sys.stderr,
        )
        return 1
    else:
        print("Starting ``ollama serve`` in the background…")
        try:
            _start_ollama_serve_background()
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"Waiting up to {args.wait:.0f}s for API…")
        if not _wait_for_server(timeout_s=args.wait):
            print(
                f"Timed out waiting for {HEALTH_URL}. "
                "Try launching **Ollama** from the Start menu, then run this script again.",
                file=sys.stderr,
            )
            return 1
        print("Ollama API is up.")

    if not args.skip_pull:
        print(f"Pulling model ``{args.model}`` (this can take several minutes)…")
        rc = _run_ollama_pull(args.model)
        if rc != 0:
            print(f"``ollama pull`` exited with code {rc}.", file=sys.stderr)
            return rc
        print("Pull complete.")

    print()
    print("--- Add or merge under ``improvement`` in config/default.yaml ---")
    print(f"""  llm_advisor_enabled: true
  llm_api_base_url: {OPENAI_BASE}
  llm_model: {args.model}
  llm_max_tokens: 2000
  llm_temperature: 0.2""")
    print()
    print("--- Environment (.env) ---")
    print("  IMPROVEMENT_LLM_API_KEY=ollama")
    print("  (Ollama ignores the value; it only needs to be non-empty for this project.)")
    print()
    print("Quick check:")
    print(f'  curl {OPENAI_BASE.replace("/v1", "")}/api/tags')
    return 0


if __name__ == "__main__":
    sys.exit(main())
