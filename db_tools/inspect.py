# db_tools/inspect.py
"""
This module provides utilities for inspecting the structure of a SQLite database, including
listing all tables and printing the schema of specific tables. It is designed to help users understand the layout of their database and the data types of each column, which is essential
for effective data analysis and manipulation.
Dependencies:
sqlite3: For interacting with SQLite databases.
pandas: For DataFrame operations and SQL query execution.
Functions:
- get_tables: Returns a list of all tables in the database.
- print_schema: Prints the schema (columns and data types) for a specific table.
Usage:
    >>> get_tables('my_database.db')
    ['table1', 'table2']
    >>> print_schema('my_database.db', 'table1')
    --- Schema for 'table1' ---
    column1  TEXT
    column2  INTEGER
    ------------------------------"""
import sqlite3
import pandas as pd

def get_tables(db_path):
    """Returns a list of all tables in the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables_df = pd.read_sql_query(query, conn)
            return tables_df['name'].tolist()
    except Exception as e:
        print(f"Error reading tables: {e}")
        return []

def print_schema(db_path, table_name):
    """Prints the schema (columns and data types) for a specific table."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Notice the double quotes around table_name to handle spaces!
            query = f'PRAGMA table_info("{table_name}");'
            schema_df = pd.read_sql_query(query, conn)
            
            if schema_df.empty:
                print(f"Table '{table_name}' not found or has no columns.")
            else:
                print(f"--- Schema for '{table_name}' ---")
                print(schema_df[['name', 'type']].to_string(index=False))
                print("-" * 30)
    except Exception as e:
        print(f"Error reading schema: {e}")