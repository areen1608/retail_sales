import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Retail Sales Analysis", layout="wide")
st.title("📊 Retail Sales Data Analysis Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("retail_sales_dataset.csv")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(inplace=True)
    bins = [0, 18, 25, 35, 50, 100]
    labels = ['<18', '18-25', '26-35', '36-50', '50+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

df = load_data()

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filter Data")
gender_filter = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
age_filter = st.sidebar.multiselect("Select Age Group", options=df["Age Group"].unique(), default=df["Age Group"].unique())

df_filtered = df[(df["Gender"].isin(gender_filter)) & (df["Age Group"].isin(age_filter))]

# -------------------------
# EDA - SALES OVER TIME
# -------------------------
st.subheader("📈 Sales Over Time")
sales_over_time = df_filtered.groupby('Date')['Total Amount'].sum()
fig1, ax1 = plt.subplots(figsize=(10, 4))
sales_over_time.plot(ax=ax1)
ax1.set_title("Sales Over Time")
ax1.set_ylabel("Total Sales Amount")
st.pyplot(fig1)

# -------------------------
# GENDER-WISE SALES
# -------------------------
st.subheader("🧍 Gender-wise Sales")
fig2, ax2 = plt.subplots()
sns.barplot(x='Gender', y='Total Amount', data=df_filtered, estimator=sum, ax=ax2)
ax2.set_title("Total Sales by Gender")
st.pyplot(fig2)

# -------------------------
# AGE GROUP SALES
# -------------------------
st.subheader("📊 Sales by Age Group")
fig3, ax3 = plt.subplots()
sns.barplot(x='Age Group', y='Total Amount', data=df_filtered, estimator=sum, order=['<18', '18-25', '26-35', '36-50', '50+'], ax=ax3)
ax3.set_title("Sales by Age Group")
st.pyplot(fig3)

# -------------------------
# TOP PRODUCT CATEGORIES
# -------------------------
st.subheader("🏆 Top Product Categories")
top_products = df_filtered.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).head(10)
fig4, ax4 = plt.subplots()
top_products.plot(kind='bar', ax=ax4)
ax4.set_title("Top Product Categories by Sales")
ax4.set_ylabel("Total Sales Amount")
st.pyplot(fig4)

# -------------------------
# PREDICTIVE MODEL (OPTIONAL)
# -------------------------
st.subheader("🤖 Predictive Model - Sales Amount")
df_encoded = pd.get_dummies(df_filtered, columns=['Gender', 'Product Category', 'Age Group'], drop_first=True)
X = df_encoded.drop(['Transaction ID', 'Customer ID', 'Date', 'Total Amount'], axis=1)
y = df_encoded['Total Amount']

if len(X) > 0: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"*Mean Squared Error:* {mean_squared_error(y_test, y_pred):.2f}")
else:
    st.warning("Not enough data after applying filters for model training.")

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("📌 Key Insights")
if not df_filtered.empty:
    st.write(f"*Highest sales from:* {top_products.index[0]}")
    st.write(f"*Top spending age group:* {df_filtered.groupby('Age Group')['Total Amount'].sum().idxmax()}")
    st.write(f"*Gender with highest sales:* {df_filtered.groupby('Gender')['Total Amount'].sum().idxmax()}")
else:
    st.warning("No data available for insights with current filters.")