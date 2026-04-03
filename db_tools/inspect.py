# db_tools/inspect.py
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