def create_filter_radio(df, column, container, label=None, include_all=True):
    """
    Create a radio button for filtering based on a column.

    Args:
        df: DataFrame containing the data
        column: Column name to filter on
        container: Streamlit container to place the radio button in
        label: Label for the radio button (defaults to "Filter by {column}")
        include_all: Whether to include an "All" option (defaults to True)

    Returns:
        Selected value from the radio button
    """
    if label is None:
        label = f"Filter by {column}"

    selected_value = "All" if include_all else None
    if column in df.columns:
        values = df[column].dropna().unique().tolist()
        try:
            values.sort(key=lambda x: int(x))
        except ValueError:
            values.sort()
        if values:
            if include_all:
                selected_value = container.radio(label, ["All"] + values)
            else:
                # Default to the first value when "All" is not included
                selected_value = container.radio(label, values, index=0)

    return selected_value


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
