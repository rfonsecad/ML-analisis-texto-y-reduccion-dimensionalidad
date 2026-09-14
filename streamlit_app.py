
import importlib
import platform
import sys
import tempfile

import streamlit as st

st.set_page_config(page_title="Prueba de compatibilidad")
st.title("Prueba de compatibilidad de librerías")
st.caption(f"Python {platform.python_version()} · {sys.platform}")

# ---------------------------------------------------------------- versiones
LIBS = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "nltk": "nltk",
    "scikit-learn": "sklearn",
    "wordcloud": "wordcloud",
    "joblib": "joblib",
}

st.subheader("1. Importación y versiones")
rows = []
for nombre, modulo in LIBS.items():
    try:
        mod = importlib.import_module(modulo)
        rows.append({"librería": nombre, "versión": getattr(mod, "__version__", "?"), "estado": "✅ OK"})
    except Exception as exc:  # noqa: BLE001
        rows.append({"librería": nombre, "versión": "-", "estado": f"❌ {exc}"})

import pandas as pd  # noqa: E402

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

