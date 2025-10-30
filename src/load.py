import pandas as pd
from pathlib import Path
from typing import List, Optional


def load_csv(
    file_path: str, column_names: Optional[List[str]] = None, delimiter: str = ","
) -> pd.DataFrame:
    """
    Load a CSV/TXT file into a pandas DataFrame.
    Assumes no header in the file and sets column names in code.

    Args:
        file_path (str): Path to the CSV/TXT file
        column_names (Optional[List[str]]): List of column names to assign
        delimiter (str): Field delimiter (default: ',')

    Returns:
        pd.DataFrame: Loaded DataFrame with assigned column names
    """
    # Load file without header
    df = pd.read_csv(file_path, header=None, delimiter=delimiter)

    # Set column names if provided
    if column_names:
        if len(column_names) != len(df.columns):
            raise ValueError(
                f"Number of column names ({len(column_names)}) doesn't match number of columns in file ({len(df.columns)})"
            )
        df.columns = column_names
    else:
        # Default column names if none provided
        df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]

    print(f"✅ Loaded file: {Path(file_path).name}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")

    return df


def load_sav(
    file_path, usecols=None, convert_categoricals: bool = False
) -> pd.DataFrame:
    df = pd.read_spss(file_path, usecols, convert_categoricals)

    return df
