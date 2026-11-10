"""
NutriCheck — Root Streamlit Entrypoint.
Delegates to person3-nlp/streamlit_app.py so running streamlit from either the project
root or the person3-nlp folder works seamlessly.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PERSON3 = ROOT / "person3-nlp"

for p in (PERSON3 / "src", PERSON3, ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Execute the main single-page dashboard
app_file = PERSON3 / "streamlit_app.py"
with open(app_file, encoding="utf-8") as f:
    code = compile(f.read(), str(app_file), "exec")
    exec(code, globals())

