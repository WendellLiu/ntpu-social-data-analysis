import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def create_mock_dataframe(num_rows: int = 50) -> pd.DataFrame:
    """
    Generate a mock DataFrame with sample data for testing purposes.

    Args:
        num_rows (int): Number of rows to generate

    Returns:
        pd.DataFrame: Mock DataFrame with various data types
    """
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)

    # Sample data lists
    names = [
        "Alice Johnson",
        "Bob Smith",
        "Charlie Brown",
        "Diana Prince",
        "Eve Wilson",
        "Frank Miller",
        "Grace Lee",
        "Henry Davis",
        "Ivy Chen",
        "Jack Taylor",
    ]

    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]

    cities = ["Taipei", "Kaohsiung", "Taichung", "Tainan", "Hsinchu", "Keelung"]

    # Generate mock data
    data = {
        "ID": range(1, num_rows + 1),
        "Name": [random.choice(names) for _ in range(num_rows)],
        "Department": [random.choice(departments) for _ in range(num_rows)],
        "Salary": np.random.normal(75000, 15000, num_rows).round(2),
        "Age": np.random.randint(22, 65, num_rows),
        "City": [random.choice(cities) for _ in range(num_rows)],
        "Join_Date": [
            datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            for _ in range(num_rows)
        ],
        "Performance_Score": np.random.uniform(1.0, 5.0, num_rows).round(2),
        "Is_Remote": np.random.choice([True, False], num_rows, p=[0.3, 0.7]),
        "Projects_Completed": np.random.poisson(8, num_rows),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Format salary as currency
    df["Salary"] = df["Salary"].apply(lambda x: f"${x:,.2f}")

    # Format join date
    df["Join_Date"] = df["Join_Date"].dt.strftime("%Y-%m-%d")

    return df
