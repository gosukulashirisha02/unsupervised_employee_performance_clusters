import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("👨‍💼 Employee Performance Clustering")
layout="wide"

st.markdown("""
<style>

/* Full page light brown background */
.stApp {
    background-color: #F5E6D3;
}

/* Use more of the page width and add top spacing */
.block-container {
    max-width: 1200px;
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Main title */
h1 {
    color: #4A2C1A;
    text-align: center;
    font-size: 42px;
    white-space: nowrap;
    margin-bottom: 50px;
}

/* Input labels */
.stNumberInput label {
    color: #7A4B2A !important;
    font-size: 17px;
    font-weight: 600;
}

/* Button */
.stButton > button {
    background-color: #8B5A2B;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-size: 16px;
}

.stButton > button:hover {
    background-color: #6F451F;
    color: white;
}

</style>
""", unsafe_allow_html=True)

employee_performance=pd.read_csv("ML_Report.csv")

# st.title("👨‍💼 Employee Performance Clustering")

experience = st.number_input(
    "Enter Experience Years",
    min_value=0.0,
    max_value=50.0,
    value=2.0
)
salary = st.number_input(
    "Enter Monthly Salary (₹)",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)

X=employee_performance[['Experience_Years','Monthly_Salary']].copy()

# X = X.fillna(0)
X['Monthly_Salary'] = X['Monthly_Salary'] / 1000

X = X[X['Monthly_Salary'] <= 200]

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

X['group'] = model.fit_predict(X)

if st.button("Run Clustering"):

    # Convert new employee salary into thousands
    new_employee = [[experience, salary / 1000]]

    # Predict cluster
    predicted_cluster = model.predict(new_employee)[0]

    st.success(
        f"The employee belongs to Cluster {predicted_cluster}"
    )

    plt.figure(figsize=(7, 4))

    scatter = plt.scatter(
        X['Experience_Years'],
        X['Monthly_Salary'],
        c=X['group'],
        cmap='viridis',
        s=40
    )

    # Display new employee
    plt.scatter(
        experience,
        salary / 1000,
        color='red',
        marker='*',
        s=250,
        label='New Employee'
    )

    plt.xlabel("Experience Years")
    plt.ylabel("Monthly Salary (Thousands)")
    plt.title("Employee Performance Clusters")
    plt.legend()

    plt.colorbar(scatter, label="Group")

    st.pyplot(plt.gcf())

    plt.close()