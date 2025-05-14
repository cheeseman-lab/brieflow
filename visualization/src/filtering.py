import streamlit as st

def create_filter_radio(df, column, container, label=None, include_all=True, key=None):
    """
    Create a radio button for filtering based on a column.

    Args:
        df: DataFrame containing the data
        column: Column name to filter on
        container: Streamlit container to place the radio button in
        label: Label for the radio button (defaults to "Filter by {column}")
        include_all: Whether to include an "All" option (defaults to True)
        key: Unique key for the Streamlit radio widget (defaults to None)

    Returns:
        Selected value from the radio button
    """
    if label is None:
        label = f"Filter by {column}"

    # Initialize the selection in session state if it doesn't exist
    state_key = f"{key}_selection"
    if state_key not in st.session_state:
        st.session_state[state_key] = "All" if include_all else None

    def on_change():
        st.session_state[state_key] = st.session_state[key]

    if column in df.columns:
        values = df[column].dropna().unique().tolist()
        try:
            values.sort(key=lambda x: int(x))
        except ValueError:
            values.sort()
        if values:
            options = ["All"] + values if include_all else values
            # Find the index of the current selection
            index = 0
            if st.session_state[state_key] in options:
                index = options.index(st.session_state[state_key])
            
            # Create the radio button with the callback
            container.radio(
                label,
                options,
                index=index,
                key=key,
                on_change=on_change
            )

    return st.session_state[state_key]


def apply_filter(df, column, selected_value):
    """
    Apply a filter to the dataframe based on the selected value.

    Args:
        df: DataFrame to filter
        column: Column name to filter on
        selected_value: Value to filter for

    Returns:
        Filtered DataFrame
    """
    if selected_value != "All" and selected_value is not None and column in df.columns:
        return df[df[column] == selected_value]
    return df
