import streamlit as st
import git
import os
from src.config import CONFIG_PATH

st.set_page_config(
    page_title="Analysis Overview - Brieflow Analysis",
    layout="wide",
)

def display_yaml(file_path):
    with open(file_path, 'r') as file:
        st.code(file.read(), language="yaml")

def display_git_info():
    # Get git repository information
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha
        
        # Check if there are any remotes
        if repo.remotes:
            # Get the first remote's URL if origin doesn't exist
            remote_url = repo.remotes[0].url if not hasattr(repo.remotes, 'origin') else repo.remotes.origin.url
            st.write(f"**Repository URL:** {remote_url}")
        else:
            st.write("**Repository URL:** No remote repositories configured")
        
        st.write(f"**Current Commit Hash:** {commit_hash}")
    except Exception as e:
        st.error(f"Error retrieving git information: {str(e)}")


def display_requirements():
    # Read and display conda environment file
    st.header("Conda Environment Configuration")
    st.write("""
    **brieflow_main_env.yml** contains the conda environment configuration for the project. 
    We use **conda** to manage dependencies:
    - The environment is defined in brieflow_main_env.yml
    - Run `conda env create -f brieflow_main_env.yml` to create the environment
    - Run `conda activate brieflow_main_env` to activate the environment
    """)
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../brieflow_main_env.yml")
        with open(env_path, 'r') as file:
            env_content = file.read()

        st.code(env_content, language="yaml")
    except Exception as e:
        st.error(f"Error reading conda environment file: {str(e)}")



st.title("Analysis Overview")

# tabs for: config, git, dependencies
tab1, tab2, tab3 = st.tabs(["Config", "Git", "Dependencies"])
with tab1:
    st.header("Screen Configuration")
    display_yaml(CONFIG_PATH)

with tab2:
    display_git_info()

with tab3:
    display_requirements()
