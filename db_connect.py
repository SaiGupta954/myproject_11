import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

# Define the connection details
server = 'newretailserver123.database.windows.net'  # Replace with your server name
database = 'RetailDB'  # Replace with your database name
username = 'azureuser'  # Replace with your Azure username
password = 'YourStrongP@ssw0rd'  # Replace with your password
driver = 'ODBC Driver 18 for SQL Server'  # Use the correct ODBC driver

# Create the connection string
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Create an engine to connect to the database
engine = create_engine(connection_string)

# Test the connection by fetching some data
try:
    # Query the first 10 rows of the 'households' table (adjust table name as needed)
    query = "SELECT TOP 10 * FROM dbo.households"
    df = pd.read_sql(query, engine)

    # If the query succeeds, display the results
    print("Data fetched successfully:")
    print(df.head())  # Show the first 5 rows of the result

except Exception as e:
    print(f"Error: {e}")

