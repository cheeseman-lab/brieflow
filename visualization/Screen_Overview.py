import streamlit as st
from pathlib import Path
import yaml
import os

SCREEN_PATH = os.getenv('BRIEFLOW_SCREEN_PATH', '../../screen.yaml')

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def display_yaml_as_json(file_path):
    try:
        data = load_yaml(file_path)
        for key, value in data.items():
            st.header(key.capitalize())
            st.json(value)
    except Exception as e:
        st.error(f"Error loading YAML: {str(e)}")

st.title("Screen Configuration")

def display_yaml(file_path):
    with open(file_path, 'r') as file:
        st.code(file.read(), language="yaml")

display_yaml(SCREEN_PATH)

