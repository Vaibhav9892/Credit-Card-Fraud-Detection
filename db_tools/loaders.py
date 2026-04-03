# db_tools/loaders.py
import sqlite3
import pandas as pd

def load_table_to_df(db_path, table_name):
    """Loads an entire table into a Pandas DataFrame."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Double quotes handle table names with spaces
            query = f'SELECT * FROM "{table_name}"'
            print(f"Loading full table '{table_name}'...")
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading table: {e}")
        return None

def query_to_df(db_path, query):
    """Executes a custom SQL query and returns the results as a DataFrame."""
    try:
        with sqlite3.connect(db_path) as conn:
            print("Executing custom query...")
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None