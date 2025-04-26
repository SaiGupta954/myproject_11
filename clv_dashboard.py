import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pyodbc
from datetime import datetime
import hashlib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# Helper functions for password hashing
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

st.set_page_config(page_title="🍭️ Retail Insights Dashboard", layout="wide")

if 'user_db' not in st.session_state:
    st.session_state.user_db = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def login_signup():
    if not st.session_state.authenticated:
        auth_option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])
        if auth_option == "Signup":
            st.sidebar.subheader("Create Account")
            new_user = st.sidebar.text_input("Username")
            new_email = st.sidebar.text_input("Email")
            new_password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Signup"):
                if new_user and new_password:
                    if new_user in st.session_state.user_db:
                        st.sidebar.error("Username already exists.")
                    else:
                        hashed_pw = make_hashes(new_password)
                        st.session_state.user_db[new_user] = {"email": new_email, "password": hashed_pw}
                        st.sidebar.success("Signup successful. Please login.")
                else:
                    st.sidebar.error("Username and password cannot be empty.")
        elif auth_option == "Login":
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Login"):
                user = st.session_state.user_db.get(username)
                if user and check_hashes(password, user['password']):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")

login_signup()

if st.session_state.authenticated:
    st.title("📊 Retail Customer Analytics Dashboard")

    @st.cache_data(ttl=600)
    def load_data():
        server = 'newretailserver123.database.windows.net'
        database = 'RetailDB'
        username = 'azureuser'
        password = 'YourStrongP@ssw0rd'
        driver = '{ODBC Driver 18 for SQL Server}'
        conn_str = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            'Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
        )
        conn = pyodbc.connect(conn_str)
        query_transactions = "SELECT * FROM Transactions"
        query_households = "SELECT * FROM Households"
        query_products = "SELECT * FROM Products"
        df_transactions = pd.read_sql(query_transactions, conn)
        df_households = pd.read_sql(query_households, conn)
        df_products = pd.read_sql(query_products, conn)
        conn.close()
        df_transactions.columns = df_transactions.columns.str.strip()
        df_households.columns = df_households.columns.str.strip()
        df_products.columns = df_products.columns.str.strip()
        return df_transactions, df_households, df_products

    df_transactions, df_households, df_products = load_data()

    df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
    df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
    df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

    full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
    full_df = full_df.merge(df_products, on='product_num', how='left')

    st.header("📈 Customer Engagement Over Time")
    df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')
    weekly_engagement = df_transactions.groupby(df_transactions['date'].dt.to_period('W'))['SPEND'].sum().reset_index()
    weekly_engagement['ds'] = weekly_engagement['date'].dt.start_time
    st.line_chart(weekly_engagement.set_index('ds')['SPEND'])

    st.header("👨👩👧 Demographics and Engagement")
    selected_demo = st.selectbox("Segment by:", ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'])
    demo_spending = full_df.groupby(selected_demo)['SPEND'].sum().reset_index()
    st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))

    st.header("🔍 Customer Segmentation")
    segmentation = full_df.groupby(['hshd_num']).agg({'SPEND': 'sum', 'INCOME_RANGE': 'first', 'AGE_RANGE': 'first'})
    st.dataframe(segmentation.sort_values(by='SPEND', ascending=False).head(10))

    st.header("🌟 Loyalty Program Effect")
    if 'LOYALTY_FLAG' in df_households.columns:
        loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
        st.dataframe(loyalty)

    st.header("🧺 Basket Analysis")
    basket = df_transactions.groupby(['BASKET_NUM', 'product_num'])['SPEND'].sum().reset_index()
    top_products = basket.groupby('product_num')['SPEND'].sum().nlargest(10).reset_index()
    top_products = top_products.merge(df_products, on='product_num', how='left')
    if 'COMMODITY' in top_products.columns:
        st.bar_chart(top_products.set_index('COMMODITY')['SPEND'])
        product_spending = top_products.groupby('COMMODITY')['SPEND'].sum().reset_index()
        fig = px.pie(product_spending, values='SPEND', names='COMMODITY', title='Spending Distribution by Product Category')
        st.plotly_chart(fig)
    else:
        st.dataframe(top_products)

    st.header("📆 Seasonal Spending Patterns")
    df_transactions['month'] = df_transactions['date'].dt.month_name()
    seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
    seasonal['month'] = pd.Categorical(seasonal['month'], categories=[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ], ordered=True)
    seasonal = seasonal.sort_values('month')
    st.bar_chart(seasonal.set_index('month'))

    st.header("💰 Customer Lifetime Value")
    clv = df_transactions.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
    st.dataframe(clv.head(10))

   

    st.header("📊 Customer Spending by Product Category")
    category_spending = full_df.groupby('COMMODITY')['SPEND'].sum().reset_index()
    st.bar_chart(category_spending.set_index('COMMODITY')['SPEND'])

    st.header("🏆 Top 10 Customers by Spending")
    top_customers = full_df.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
    st.dataframe(top_customers.head(10))

    st.header("📈 Trends in Age Group Spending")
    age_group_spending = full_df.groupby('AGE_RANGE')['SPEND'].sum().reset_index()
    st.bar_chart(age_group_spending.set_index('AGE_RANGE')['SPEND'])
    fig = px.pie(age_group_spending, values='SPEND', names='AGE_RANGE', title='Spending Distribution by Age Group')
    st.plotly_chart(fig)

    import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# ... (your data loading and merging code here, as in clv_dashboard.py) ...
if st.session_state.authenticated:
    st.title("📊 Retail Customer Analytics Dashboard")

    # --- Load and prepare data ---
    df_transactions, df_households, df_products = load_data()

    df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
    df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
    df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

    full_df = df_transactions.merge(df_households, on='hshd_num', how='left')
    full_df = full_df.merge(df_products, on='product_num', how='left')
    df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w')

    # --- Tabs for ML Analysis ---
    tab1, tab2 = st.tabs(["⚠️ Churn Prediction", "🧺 Basket Analysis"])

    with tab1:
        st.header("Churn Prediction: Customer Engagement Over Time")

        # Filter options
        departments = ["All"] + sorted(df_products['DEPARTMENT'].dropna().unique())
        commodities = ["All"] + sorted(df_products['COMMODITY'].dropna().unique())
        brand_types = ["All"] + sorted(df_products['BRAND_TY'].dropna().unique())
        organic_flags = ["All"] + sorted(df_products['NATURAL_ORGANIC_FLAG'].dropna().unique())

        col1, col2, col3, col4 = st.columns(4)
        department = col1.selectbox("Select Department:", departments)
        commodity = col2.selectbox("Select Commodity:", commodities)
        brand_type = col3.selectbox("Select Brand Type:", brand_types)
        organic_flag = col4.selectbox("Select Organic:", organic_flags)

        if st.button("Apply Filters", key="churn_apply"):
            filtered = df_transactions.merge(df_products, on='product_num', how='left')

            if department != "All":
                filtered = filtered[filtered['DEPARTMENT'] == department]
            if commodity != "All":
                filtered = filtered[filtered['COMMODITY'] == commodity]
            if brand_type != "All":
                filtered = filtered[filtered['BRAND_TY'] == brand_type]
            if organic_flag != "All":
                filtered = filtered[filtered['NATURAL_ORGANIC_FLAG'] == organic_flag]

            if not filtered.empty:
                churn_df = filtered.groupby('date')['SPEND'].sum().reset_index().sort_values('date')
                st.line_chart(churn_df.set_index("date")["SPEND"])
                with st.expander("📄 Raw Data"):
                    st.dataframe(churn_df)
            else:
                st.warning("No data found for selected filters.")

        # ML Churn Prediction
        st.subheader("ML Churn Prediction: At-Risk Customers")
        now = df_transactions['date'].max()
        rfm = df_transactions.groupby('hshd_num').agg(
            recency=('date', lambda x: (now - x.max()).days),
            frequency=('BASKET_NUM', 'nunique'),
            monetary=('SPEND', 'sum')
        )
        rfm['churn'] = (rfm['recency'] > 84).astype(int)

        X = rfm[['recency', 'frequency', 'monetary']]
        y = rfm['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                  columns=['Predicted Not Churn', 'Predicted Churn'],
                                  index=['Actual Not Churn', 'Actual Churn']))

        feat_imp = pd.Series(clf.feature_importances_, index=['Recency', 'Frequency', 'Monetary'])
        st.write("**Feature Importances:**")
        st.bar_chart(feat_imp)

        with tab2:
            st.header("Basket Analysis - Predicting Total Spend")

    basket_merged = df_transactions.merge(df_products, on='product_num', how='left')
    basket_features = pd.get_dummies(
        basket_merged[['BASKET_NUM', 'DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']]
    )
    X_basket = basket_features.groupby('BASKET_NUM').sum()
    y_basket = basket_merged.groupby('BASKET_NUM')['SPEND'].sum()

    X_train, X_test, y_train, y_test = train_test_split(X_basket, y_basket, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**MSE:** {mse:.2f}")

    st.subheader("Predicted vs. Actual Basket Spend (Test Set)")
    chart_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.line_chart(chart_df.reset_index(drop=True))

    # ✅ Add this section (no extra indent!)
    st.subheader("📄 Actual vs. Predicted Spend Table")
    st.dataframe(chart_df)

    csv = chart_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='predicted_vs_actual_basket_spend.csv',
        mime='text/csv',
    )

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X_basket.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    st.write("**Top Drivers of Basket Spend:**")
    st.bar_chart(top_features)

