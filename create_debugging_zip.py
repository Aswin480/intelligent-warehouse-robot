"""Bundle clean source code and debugging assets into submission_debugging/Advanced_RL_Debugging_Source.zip."""
from __future__ import annotations

import os
import zipfile

ZIP_PATH = "submission_debugging/Advanced_RL_Debugging_Source.zip"

INCLUDE_PATHS = [
    "debugging_experiment",
    "warehouse_rl",
    "models",
    "tests",
    "streamlit_app.py",
    "train_agent.py",
    "evaluate_agent.py",
    "generate_debugging_pdf.py",
    "requirements.txt",
    "README.md",
    "pyproject.toml",
    os.path.join("submission_debugging", "results"),
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".venv",
    ".idea",
    ".vscode",
    ".log",
]


def make_debugging_zip():
    os.makedirs(os.path.dirname(os.path.abspath(ZIP_PATH)), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE_PATHS:
            if not os.path.exists(item):
                continue
            if os.path.isfile(item):
                zf.write(item, arcname=item)
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if not any(pat in d for pat in EXCLUDE_PATTERNS)]
                    for file in files:
                        if any(pat in file for pat in EXCLUDE_PATTERNS):
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        zf.write(full_path, arcname=rel_path)

    print(f"Created submission ZIP: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)} bytes)")


if __name__ == "__main__":
    make_debugging_zip()
