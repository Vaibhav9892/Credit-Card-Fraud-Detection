# db_tools/loaders.py
"""
This module provides utilities for loading data from a SQLite database into pandas DataFrames.
 It includes functions for loading entire tables as well as executing custom SQL queries to retrieve specific subsets of data
 . It is designed to be flexible and user-friendly, allowing for easy integration into data analysis workflows.
 Dependencies:
sqlite3: For interacting with SQLite databases.
pandas: For DataFrame operations and SQL query execution.
Functions:
- load_table_to_df: Loads an entire table into a Pandas DataFrame.
- query_to_df: Executes a custom SQL query and returns the results as a DataFrame.
Usage:
    >>> df = load_table_to_df('my_database.db', 'my_table')
    >>> query = 'SELECT column1, column2 FROM my_table WHERE column3 > 100;'
    >>> df_subset = query_to_df('my_database.db', query)
"""
import sqlite3
import pandas as pd

def load_table_to_df(db_path, table_name):
    """Loads an entire table into a Pandas DataFrame.
    Note: This can be very memory-intensive for large tables! Use with caution.
    """
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
    """Executes a custom SQL query and returns the results as a DataFrame.
     Note: Be careful with complex queries that return large datasets, as they can consume a lot of memory!
    """
    try:
        with sqlite3.connect(db_path) as conn:
            print("Executing custom query...")
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None