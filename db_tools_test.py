from db_tools import inspect, loaders

# Update this if your db is named something else!
db_file = 'creditdatabase.db'

print("=========================================")
print("🧪 READ-ONLY DATABASE TEST")
print("=========================================\n")

# --- Test 1: Inspection ---
print("--- 1. Testing Inspection Module ---")
tables = inspect.get_tables(db_file)
print(f"Tables found: {tables}")

if tables:
    first_table = tables[0]
    print(f"\nPrinting schema for '{first_table}':")
    inspect.print_schema(db_file, first_table)
    
    # --- Test 2: Loading (Safe Limit) ---
    print("\n--- 2. Testing Data Loader (Safe Mode) ---")
    # We use LIMIT 5 so it executes instantly and doesn't stress your RAM
    safe_query = f'SELECT * FROM "{first_table}" LIMIT 5;'
    
    df_test = loaders.query_to_df(db_file, safe_query)
    
    if df_test is not None:
        print(f"✅ Success! Loaded {len(df_test)} rows.")
        print("\nData Preview:")
        # Print just the first few columns so it doesn't flood your terminal
        print(df_test[['Time', 'V1', 'V2', 'Amount', 'Class']])
        print("\n🎉 All read-only modules are working perfectly!")
else:
    print("❌ No tables found. Make sure you rebuilt the database!")