import pandas as pd
import sqlite3

df = pd.read_csv("creditcard.csv")

conn = sqlite3.connect("creditdatabase.db")

df.to_sql("credit card transactions", conn, if_exists= "replace", index = False)

conn.close()