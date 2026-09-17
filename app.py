from PIL import Image

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

st.set_page_config(page_title="Mi App", layout="wide")
logo = Image.open("logo.jpg")
st.image(logo,caption=None,use_container_width=True)

logo = Image.open("ods.png")
st.image(logo,caption=None,use_container_width=True)

##############################################################################


# Diccionario oficial de metas y nombres de los ODS
ODS_INFO = {
    1: {"nombre": "Fin de la pobreza", "color": "#E6002D"},
    2: {"nombre": "Hambre cero", "color": "#D29E36"},
    3: {"nombre": "Salud y bienestar", "color": "#379B4E"},
    4: {"nombre": "Educación de calidad", "color": "#BF1332"},
    5: {"nombre": "Igualdad de género", "color": "#EB382D"},
    6: {"nombre": "Agua limpia y saneamiento", "color": "#20B0D7"},
    7: {"nombre": "Energía asequible y no contaminante", "color": "#FBB42F"},
    8: {"nombre": "Trabajo decente y crecimiento económico", "color": "#8C1137"},
    9: {"nombre": "Industria, innovación e infraestructura", "color": "#EF692C"},
    10: {"nombre": "Reducción de las desigualdades", "color": "#DC0081"},
    11: {"nombre": "Ciudades y comunidades sostenibles", "color": "#F69A34"},
    12: {"nombre": "Producción y consumo responsables", "color": "#CD8B34"},
    13: {"nombre": "Acción por el clima", "color": "#4C7742"},
    14: {"nombre": "Vida submarina", "color": "#0E7FBA"},
    15: {"nombre": "Vida de ecosistemas terrestres", "color": "#4BB051"},
    16: {"nombre": "Paz, justicia e instituciones sólidas", "color": "#065789"},
    17: {"nombre": "Alianzas para lograr los objetivos", "color": "#1A3A68"}
}

def limpiar_texto(texto):
    """Aplica la misma normalización usada en el entrenamiento."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

@st.cache_resource
def cargar_modelo():
    """Carga el pipeline serializado en caché."""
    return joblib.load('modelo_ods_pipeline.joblib')

# Configuración de página
st.set_page_config(
    page_title="Clasificador ODS - Agenda 2030",
    layout="centered"
)

st.title("Clasificador de Textos según los ODS")
st.markdown(
      'Esta herramienta procesa textos libres de planeación y políticas públicas mediante **NLP (TF-IDF + LSA)** y **Machine Learning** para identificar su alineación con los <a href="https://www.un.org/sustainabledevelopment/es/" target="_blank">Objetivos de Desarrollo Sostenible (Agenda 2030)</a>',
      unsafe_allow_html=True,
  )

try:
    pipeline = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró el archivo 'modelo_ods_pipeline.joblib'. Ejecuta primero la celda de exportación en tu notebook.")
    st.stop()

# Área de entrada
texto_input = st.text_area(
    "Ingresa el texto a evaluar:",
    height=180,
    placeholder="Escribe el texto a clasificar..."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    analizar = st.button("Clasificar Texto", type="secondary", use_container_width=True)

if analizar:
    if not texto_input.strip():
        st.warning("Por favor, ingresa un texto antes de presionar 'Clasificar'.")
    else:
        # Preprocesamiento idéntico al pipeline
        texto_limpio = limpiar_texto(texto_input)
        
        # Inferencia
        pred_ods = int(pipeline.predict([texto_limpio])[0])
        info = ODS_INFO.get(pred_ods, {"nombre": "Desconocido", "color": "#333333"})
        
        # Probabilidades si el clasificador las soporta
        tiene_proba = hasattr(pipeline.named_steps['clf'], 'predict_proba')
        
        st.markdown("---")
        st.subheader("Resultado de la Clasificación")
        
        # Tarjeta visual con color representativo del ODS
        st.markdown(
            f"""
            <div style="background-color: {info['color']}; padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h2 style="margin:0; color: white;"> ODS {pred_ods}: {info['nombre']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Desglose de probabilidades top 3
        if tiene_proba:
            probas = pipeline.predict_proba([texto_limpio])[0]
            clases = pipeline.classes_
            df_probas = pd.DataFrame({
                "ODS": [f"ODS {c}: {ODS_INFO.get(c, {}).get('nombre', '')}" for c in clases],
                "Probabilidad": probas
            }).sort_values(by="Probabilidad", ascending=False).reset_index(drop=True)
            
            st.write("**Top 3 ODS más afines:**")
            for i in range(min(3, len(df_probas))):
                fila = df_probas.iloc[i]
                st.progress(float(fila["Probabilidad"]), text=f"{fila['ODS']} ({fila['Probabilidad']*100:.1f}%)")

###############################################################################

# --- CÓDIGO DEL FOOTER ---
footer_html = """
<style>
.footer {
    
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #003c6e;
    color: #fff;
    text-align: center;
    padding: 10px;
    font-size: 18px;
    border-top: 1px solid #e7e7e7;
    z-index: 100;
}
/* Ajuste para modo oscuro de Streamlit (opcional) */
@media (prefers-color-scheme: dark) {
    .footer {
        background-color: #0e1117;
        color: #ffffff;
        border-top: 1px solid #262730;
    }
}
</style>
<div class="footer">
    <p>Desarrollado por Javier Obando y Luis Orozco | Universidad de los Andes | © 2026</p>
</div>
"""

# Renderizar el footer
st.markdown(footer_html, unsafe_allow_html=True)