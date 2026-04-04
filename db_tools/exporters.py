# db_tools/exporters.py

'''
This module provides utilities for exporting data from pandas DataFrames to various formats,
including CSV files and SQLite databases. It is designed to handle common data export tasks
in data science and machine learning workflows, particularly for saving processed datasets
and query results.

Dependencies:

sqlite3: For interacting with SQLite databases.
pandas: For DataFrame operations and SQL query execution.

Functions:
- df_to_csv: Exports a Pandas DataFrame to a CSV file.
- query_to_csv: Executes a query and exports the direct result to a CSV.
- create_xy_database: Creates a new SQLite database containing two tables (X and Y).

Usage:
    >>> df_to_csv(my_dataframe, 'output.csv')
    >>> query_to_csv('my_database.db', 'SELECT * FROM my_table;', 'query_output.csv')   
    >>> create_xy_database('xy_database.db', df_X, df_Y)
'''
import sqlite3
import pandas as pd

def df_to_csv(df, output_path):
    """Exports a Pandas DataFrame to a CSV file."""
    try:
        print(f"Exporting DataFrame to {output_path}...")
        # index=False prevents pandas from writing the row numbers as a separate column
        df.to_csv(output_path, index=False)
        print("Export complete!")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

def query_to_csv(db_path, query, output_path):
    """Executes a query and exports the direct result to a CSV."""
    try:
        with sqlite3.connect(db_path) as conn:
            print("Running query and exporting to CSV...")
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False)
            print(f"Saved directly to {output_path}")
    except Exception as e:
        print(f"Error exporting query to CSV: {e}")

def create_xy_database(new_db_path, df_X, df_Y):
    """
    Creates a brand new SQLite database containing two tables: 'X' and 'Y'.
    Useful for saving pre-processed training and testing splits.
    """
    try:
        with sqlite3.connect(new_db_path) as conn:
            print(f"Creating new database at {new_db_path}...")
            
            # Write df_X to a table named 'X'
            df_X.to_sql('X', conn, if_exists='replace', index=False)
            print("Table 'X' created successfully.")
            
            # Write df_Y to a table named 'Y'
            df_Y.to_sql('Y', conn, if_exists='replace', index=False)
            print("Table 'Y' created successfully.")
            
            print("Database creation complete!")
    except Exception as e:
        print(f"Error creating XY database: {e}")