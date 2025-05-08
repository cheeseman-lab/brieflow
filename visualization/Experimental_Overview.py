import streamlit as st
import yaml
from src.config import SCREEN_PATH

st.set_page_config(page_title="Experimental Overview - Brieflow Analysis")


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def display_yaml(file_path):
    with open(file_path, "r") as file:
        st.code(file.read(), language="yaml")


st.title("Experimental Overview")
display_yaml(SCREEN_PATH)
