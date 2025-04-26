import pyodbc
import pandas as pd
import streamlit as st

# Database connection settings
server = 'newretailserver123.database.windows.net'
database = 'RetailDB'
username = 'azureuser'
password = 'YourStrongP@ssw0rd'
driver = 'ODBC Driver 18 for SQL Server'

# Function to establish database connection
def get_connection():
    conn = pyodbc.connect('DRIVER={};SERVER={};PORT=1433;DATABASE={};UID={};PWD={}'.format(
        driver, server, database, username, password))
    return conn

# Function to load data based on HSHD_NUM
def load_data(hshd_num):
    query = """
    SELECT 
        H.HSHD_NUM, 
        T.BASKET_NUM, 
        T.PURCHASE_ AS Date,  -- Using PURCHASE_ for date
        P.PRODUCT_NUM, 
        P.DEPARTMENT, 
        P.COMMODITY 
    FROM dbo.households H
    JOIN dbo.transactions T ON H.HSHD_NUM = T.HSHD_NUM
    LEFT JOIN dbo.products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
    WHERE H.HSHD_NUM = ?
    ORDER BY H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_, P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY;
    """
    conn = get_connection()
    df = pd.read_sql(query, conn, params=[hshd_num])
    conn.close()
    return df

# Streamlit UI components
def app():
    st.title("Interactive Data Pull for Household Number (HSHD_NUM)")
    st.write("Enter Household Number to fetch associated data")

    # User input for HSHD_NUM
    hshd_num = st.text_input("Enter Household Number (HSHD_NUM):")

    if hshd_num:
        try:
            # Fetch data for the entered HSHD_NUM
            hshd_num = int(hshd_num)
            st.write(f"Fetching data for HSHD_NUM: {hshd_num}")
            data = load_data(hshd_num)

            if not data.empty:
                st.write(f"Data for HSHD_NUM {hshd_num}")
                st.dataframe(data)  # Display the fetched data in a table
            else:
                st.write("No data found for the entered Household Number.")

        except ValueError:
            st.error("Please enter a valid numeric Household Number (HSHD_NUM).")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
