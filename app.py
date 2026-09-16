from PIL import Image

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

st.set_page_config(page_title="Mi App", layout="wide")
logo = Image.open("logo.png")
st.image(logo,caption=None,use_container_width=True)
st.title("Clasificador de texto según los 17 ODS")


# Crear el text area
texto_clasificar = st.text_area(
    label="Escribe el texto a clasificar:",
    placeholder="digita el texto aquí...",
    height=150 # Altura en píxeles (opcional)
)

# 2. Crear el botón para dar el resultado
if st.button("Clasificar"):
    # Validar que el usuario haya escrito algo
    if texto_clasificar.strip():
        # Aquí colocas la lógica de tu aplicación
        resultado = texto_clasificar.upper() # Ejemplo: convertir a mayúsculas
        
        st.success("¡Clasificación exitosa!")
        st.write("### Resultado:")
        st.write(resultado)
    else:
        st.warning("Por favor, escribe el texto a clasificar.")
