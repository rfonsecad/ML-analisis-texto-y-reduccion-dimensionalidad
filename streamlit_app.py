import re

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)


def load_processors():
    stopwords_spanish = stopwords.words("spanish")
    tokenizer = RegexpTokenizer(r"[a-záéíóúüñ][a-záéíóúüñ0-9]+")
    stemmer = SnowballStemmer("spanish")
    return stopwords_spanish, tokenizer, stemmer


stopwords_spanish, tokenizer, stemmer = load_processors()


def normalizar_grados(texto):
    texto = re.sub(r"[º°]\s*c\b", " gradoscelsius ", texto)
    texto = re.sub(r"\bgrados?\s+(?:celsius|cent[íi]grados?)\b", " gradoscelsius ", texto)
    return texto


def text_preprocess(texto):
    texto = texto.lower()
    texto = normalizar_grados(texto)
    tokens = tokenizer.tokenize(texto)
    tokens = [word for word in tokens if word not in stopwords_spanish]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def text_preprocess_batch(X):
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return pd.Series(X).apply(text_preprocess)


NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
}


@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("resources/models/pipeline_tfidf.joblib")
    modelo = joblib.load("resources/models/modelo_ods.joblib")
    return pipeline, modelo


st.set_page_config(page_title="Clasificador de ODS")
st.title("Clasificador de textos según los ODS")
st.write(
    "Ingresa un texto y el modelo indicará a qué Objetivo de Desarrollo Sostenible se relaciona."
)

pipeline, modelo = cargar_modelos()

texto = st.text_area(
    "Texto a clasificar",
    height=200,
    placeholder="Ejemplo: El acceso a agua potable y saneamiento sigue siendo limitado en zonas rurales...",
)

if st.button("Predecir", type="primary"):
    if not texto.strip():
        st.warning("Escribe un texto antes de predecir.")
    else:
        X_nuevo = pipeline.transform([texto])
        prediccion = int(modelo.predict(X_nuevo)[0])
        probabilidades = modelo.predict_proba(X_nuevo)[0]

        st.success(f"**ODS {prediccion}: {NOMBRES_ODS[prediccion]}**")

        top = (
            pd.DataFrame({"ODS": modelo.classes_, "Probabilidad": probabilidades})
            .sort_values("Probabilidad", ascending=False)
            .head(3)
        )
        top["Nombre"] = top["ODS"].map(NOMBRES_ODS)
        top["Probabilidad"] = (top["Probabilidad"] * 100).round(1).astype(str) + " %"

        st.subheader("Tres ODS más probables")
        st.dataframe(top[["ODS", "Nombre", "Probabilidad"]], hide_index=True)
