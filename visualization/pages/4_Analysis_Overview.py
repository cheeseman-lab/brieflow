import streamlit as st
import git
import os
from src.config import CONFIG_PATH

st.set_page_config(
    page_title="Analysis Overview - Brieflow Analysis",
    layout="wide",
)


def display_yaml(file_path):
    with open(file_path, "r") as file:
        st.code(file.read(), language="yaml")


def display_git_info():
    # Get git repository information
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha

        # Check if there are any remotes
        if repo.remotes:
            # Get the first remote's URL if origin doesn't exist
            remote_url = (
                repo.remotes[0].url
                if not hasattr(repo.remotes, "origin")
                else repo.remotes.origin.url
            )
            st.write(f"**Repository URL:** {remote_url}")
        else:
            st.write("**Repository URL:** No remote repositories configured")

        st.write(f"**Current Commit Hash:** {commit_hash}")
    except Exception as e:
        st.error(f"Error retrieving git information: {str(e)}")


def display_requirements():
    st.header("Dependencies")
    try:
        _vis_dir = os.path.dirname(os.path.dirname(__file__))
        pyproject_path = os.path.join(os.path.dirname(_vis_dir), "pyproject.toml")
        with open(pyproject_path, "r") as file:
            content = file.read()
        st.code(content, language="toml")
    except Exception as e:
        st.error(f"Error reading pyproject.toml: {str(e)}")


st.title("Analysis Overview")

# tabs for: config, dependencies, git
tab1, tab2, tab3 = st.tabs(["Config", "Dependencies", "Git"])
with tab1:
    display_yaml(CONFIG_PATH)

with tab2:
    display_requirements()

with tab3:
    display_git_info()
