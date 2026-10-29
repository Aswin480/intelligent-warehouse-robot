"""Create clean source code ZIP for assignment submission."""
from __future__ import annotations

import os
import zipfile

ZIP_PATH = "submission/RL_Challenge_Source_Code.zip"

INCLUDE_PATHS = [
    "warehouse_rl",
    "models",
    "tests",
    "static",
    "gridworld",
    "main.py",
    "mdp.py",
    "async_dp.py",
    "streamlit_app.py",
    "train_agent.py",
    "evaluate_agent.py",
    "generate_report_pdf.py",
    "requirements.txt",
    "README.md",
    "PROJECT_REPORT.md",
    "pyproject.toml",
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


def make_zip():
    os.makedirs(os.path.dirname(os.path.abspath(ZIP_PATH)), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE_PATHS:
            if not os.path.exists(item):
                continue
            if os.path.isfile(item):
                zf.write(item, arcname=item)
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    # Filter out excluded directory patterns
                    dirs[:] = [d for d in dirs if not any(pat in d for pat in EXCLUDE_PATTERNS)]
                    for file in files:
                        if any(pat in file for pat in EXCLUDE_PATTERNS):
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ".")
                        zf.write(full_path, arcname=rel_path)

    print(f"Created submission ZIP: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)} bytes)")


if __name__ == "__main__":
    make_zip()
