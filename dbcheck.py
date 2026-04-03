import sqlite3
import pandas as pd

# Update this to your actual file path
db_path = 'creditdatabase.db' 

try:
    conn = sqlite3.connect(db_path)
    print(f"✅ Successfully connected to {db_path}\n")
    
    # ---------------------------------------------------------
    # 1. CHECK WHAT TABLES EXIST
    # ---------------------------------------------------------
    # sqlite_master contains the metadata for the whole database
    query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
    tables_df = pd.read_sql_query(query_tables, conn)
    
    if tables_df.empty:
        print("❌ WARNING: The database connected, but no tables were found inside it!")
    else:
        print("📁 TABLES FOUND IN DATABASE:")
        for index, row in tables_df.iterrows():
            print(f"  - {row['name']}")
        print("\n")
        
        # ---------------------------------------------------------
        # 2. CHECK THE SCHEMA OF THE FIRST TABLE
        # ---------------------------------------------------------
        # We will automatically grab the name of the first table we found
        first_table = tables_df['name'].iloc[0]
        
        print(f"📋 SCHEMA FOR TABLE '{first_table}':")
        # PRAGMA table_info returns the column ID, name, type, and if it can be NULL
        query_schema = f"PRAGMA table_info('{first_table}');"
        schema_df = pd.read_sql_query(query_schema, conn)
        
        # Display just the column name and data type for easy reading
        print(schema_df[['name', 'type']])

except sqlite3.OperationalError as e:
    print(f"❌ CONNECTION ERROR: Could not find or open the database. Is the path correct? \nDetails: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
finally:
    if 'conn' in locals():
        conn.close()