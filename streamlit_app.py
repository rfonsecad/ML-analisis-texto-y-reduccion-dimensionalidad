import re

import joblib
import nltk
import pandas as pd
import sklearn
import streamlit as st
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

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

# Métricas obtenidas en el notebook sobre el conjunto de test (1932 textos nunca vistos).
METRICAS = {
    "F1 macro (test)": "0.85",
    "Accuracy (test)": "0.87",
    "F1 macro (validación)": "0.81",
    "Accuracy (validación)": "0.84",
}


@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("resources/models/pipeline_tfidf.joblib")
    modelo = joblib.load("resources/models/modelo_ods.joblib")
    return pipeline, modelo


def nube_explicativa(pipeline, modelo, X_tfidf, clase):
    """Nube con los términos del texto que más empujaron la predicción hacia la clase dada.

    El clasificador es lineal sobre las componentes del SVD, así que el peso de cada término
    hacia una clase es coef_clase @ componentes_svd. La contribución de un término del texto es
    su valor TF-IDF multiplicado por ese peso. Solo se muestran contribuciones positivas.
    """
    vocab = pipeline.named_steps["vectorizacion_tfidf"].named_steps["conteo"].get_feature_names_out()
    svd, clf = modelo.named_steps["svd"], modelo.named_steps["clf"]
    idx_clase = list(clf.classes_).index(clase)
    pesos_terminos = clf.coef_[idx_clase] @ svd.components_

    fila = X_tfidf.tocsr()[0]
    contribuciones = {
        vocab[j]: float(valor * pesos_terminos[j])
        for j, valor in zip(fila.indices, fila.data)
        if valor * pesos_terminos[j] > 0
    }
    if not contribuciones:
        return None
    return WordCloud(
        width=900, height=350, background_color="white", colormap="viridis"
    ).generate_from_frequencies(contribuciones)


st.set_page_config(page_title="Clasificador de ODS", layout="wide")
st.title("Clasificador de textos según los ODS")
st.write(
    "Ingresa un texto y el modelo indicará a qué Objetivo de Desarrollo Sostenible se relaciona."
)

pipeline, modelo = cargar_modelos()
svd, clf = modelo.named_steps["svd"], modelo.named_steps["clf"]
n_vocab = len(pipeline.named_steps["vectorizacion_tfidf"].named_steps["conteo"].vocabulary_)

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Acerca del modelo")

    st.markdown(f"""
#### :green[Preprocesamiento]
- :small[Minúsculas y normalización de grados Celsius]
- :small[Tokenización con expresión regular]
- :small[Eliminación de stopwords en español]
- :small[Stemming Snowball]

#### :green[Representación]
- :small[Bolsa de palabras con pesado TF-IDF]
- :small[Vocabulario de {n_vocab:,} términos]

#### :green[Reducción de dimensionalidad]
- :small[TruncatedSVD (LSA) con {svd.n_components} componentes]

#### :green[Clasificador]
- :small[Regresión logística multinomial]
- :small[C = {clf.C} · max_iter = {clf.max_iter} · solver = {clf.solver}]
- :small[Hiperparámetros elegidos con GridSearchCV (3 pliegues, F1 macro)]
""")

    st.markdown("#### :green[Desempeño]")
    col_a, col_b = st.columns(2)
    col_a.metric("F1 macro · test", METRICAS["F1 macro (test)"])
    col_b.metric("Accuracy · test", METRICAS["Accuracy (test)"])
    col_a.metric("F1 macro · val.", METRICAS["F1 macro (validación)"])
    col_b.metric("Accuracy · val.", METRICAS["Accuracy (validación)"])

    st.markdown("""
#### :green[Datos]
- :small[OSDG Community Dataset 2023, 9.656 textos en español]
- :small[División estratificada 60 % train, 20 % validación, 20 % test]
- :small[ODS 17 no cubierto por falta de ejemplos]
- :small[ODS 8, 9 y 10 son los que más se confunden entre sí]
""")
    st.caption(f"scikit-learn {sklearn.__version__}")

# ---------------------------------------------------------------- área principal
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

        tabla = pd.DataFrame({"ODS": modelo.classes_, "Probabilidad": probabilidades})
        tabla["Nombre"] = tabla["ODS"].map(NOMBRES_ODS)

        top = tabla.sort_values("Probabilidad", ascending=False).head(3).copy()
        top["Probabilidad"] = (top["Probabilidad"] * 100).round(1).astype(str) + " %"

        st.subheader("Tres ODS más probables")
        st.dataframe(top[["ODS", "Nombre", "Probabilidad"]], hide_index=True)

        st.subheader("Probabilidad de cada ODS")
        grafica = tabla.copy()
        grafica["ODS"] = "ODS " + grafica["ODS"].astype(str).str.zfill(2)
        grafica["Probabilidad"] = (grafica["Probabilidad"] * 100).round(1)
        st.bar_chart(
            grafica, x="ODS", y="Probabilidad", horizontal=True, height=450,
            x_label="Probabilidad (%)", y_label="",
        )

        st.subheader(f"Términos del texto que apuntan al ODS {prediccion}")
        nube = nube_explicativa(pipeline, modelo, X_nuevo, prediccion)
        if nube is None:
            st.info("Ningún término del texto aporta evidencia positiva hacia este ODS.")
        else:
            st.image(nube.to_array(), width="stretch")
            st.caption(
                "El tamaño de cada término refleja cuánto contribuyó a la predicción. "
                "Los términos aparecen en su forma raíz (stem), tal como los procesa el modelo."
            )
