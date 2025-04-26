import pandas as pd
import streamlit as st
import pyodbc

# Database connection details
server = 'newretailserver123.database.windows.net'
database = 'RetailDB'
username = 'azureuser'
password = 'YourStrongP@ssw0rd'
driver = '{ODBC Driver 18 for SQL Server}'

def get_connection():
    return pyodbc.connect(
        f'DRIVER={driver};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
    )

def load_data_from_db():
    conn = get_connection()
    df_transactions = pd.read_sql("SELECT * FROM Transactions", conn)
    df_households = pd.read_sql("SELECT * FROM Households", conn)
    df_products = pd.read_sql("SELECT * FROM Products", conn)
    conn.close()
    df_transactions.columns = df_transactions.columns.str.strip()
    df_households.columns = df_households.columns.str.strip()
    df_products.columns = df_products.columns.str.strip()
    return df_transactions, df_households, df_products

st.title("Data Loader: Upload Datasets")

# --- File Uploaders ---
uploaded_transactions = st.file_uploader("Upload Transactions Dataset", type="csv")
uploaded_households = st.file_uploader("Upload Households Dataset", type="csv")
uploaded_products = st.file_uploader("Upload Products Dataset", type="csv")

# --- Handle Uploads and Session State ---
if uploaded_transactions is not None:
    st.session_state['transactions_df'] = pd.read_csv(uploaded_transactions)
if uploaded_households is not None:
    st.session_state['households_df'] = pd.read_csv(uploaded_households)
if uploaded_products is not None:
    st.session_state['products_df'] = pd.read_csv(uploaded_products)

# --- Show Data Previews if Available ---
if 'transactions_df' in st.session_state:
    st.write("Transactions Data", st.session_state['transactions_df'].head())
if 'households_df' in st.session_state:
    st.write("Households Data", st.session_state['households_df'].head())
if 'products_df' in st.session_state:
    st.write("Products Data", st.session_state['products_df'].head())

# --- Load from Database if Needed ---
if (('transactions_df' not in st.session_state) or
    ('households_df' not in st.session_state) or
    ('products_df' not in st.session_state)):
    if st.button("Load Latest Data from Database"):
        tdf, hdf, pdf = load_data_from_db()
        st.session_state['transactions_df'] = tdf
        st.session_state['households_df'] = hdf
        st.session_state['products_df'] = pdf
        st.success("Loaded latest data from database.")
        st.write("Transactions Data", tdf.head())
        st.write("Households Data", hdf.head())
        st.write("Products Data", pdf.head())
