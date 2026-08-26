import os

import streamlit as st


def load_settings():
    values = {}
    for key in ["OPENDART_API_KEY", "ECOS_API_KEY"]:
        if key in st.secrets:
            values[key] = st.secrets[key]
        elif os.getenv(key):
            values[key] = os.getenv(key)
    return values
