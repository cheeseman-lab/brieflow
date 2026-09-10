import streamlit as st
import git
import os
from src.config import CONFIG_PATHS
from src.theme import page_setup

page_setup(
    "Analysis Overview",
    "🗂️",
    "The configuration, dependencies and repository state this analysis ran with.",
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
        else:
            remote_url = "No remote repositories configured"

        st.metric("Commit", commit_hash[:12])
        st.caption("Repository")
        st.code(remote_url, language=None)
        st.caption("Full commit hash")
        st.code(commit_hash, language=None)
    except Exception as e:
        st.error(f"Error retrieving git information: {str(e)}")


def display_requirements():
    try:
        _vis_dir = os.path.dirname(os.path.dirname(__file__))
        pyproject_path = os.path.join(os.path.dirname(_vis_dir), "pyproject.toml")
        with open(pyproject_path, "r") as file:
            content = file.read()
        st.caption(f"`{os.path.basename(pyproject_path)}`")
        st.code(content, language="toml")
    except Exception as e:
        st.error(f"Error reading pyproject.toml: {str(e)}")


# tabs for: config, dependencies, git
tab1, tab2, tab3 = st.tabs(["Config", "Dependencies", "Git"])
with tab1:
    # CONFIG_PATH can name several files that the run deep-merged, so show each of them.
    for config_path in CONFIG_PATHS:
        st.subheader(os.path.basename(config_path))
        st.caption(os.path.abspath(config_path))
        display_yaml(config_path)

with tab2:
    display_requirements()

with tab3:
    display_git_info()
