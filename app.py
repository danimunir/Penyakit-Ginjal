import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def model(csv, csv2):
    # Load data
    df = pd.read_csv(csv, encoding="latin-1")
    df_new = pd.read_csv(csv2, encoding="latin-1")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    if "Unnamed: 0" in df_new.columns:
        df_new.drop("Unnamed: 0", axis=1, inplace=True)

    X = df.drop("klasifikasi", axis=1)
    y = df["klasifikasi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Load model
    models = joblib.load("./models.joblib")
    numeric = [i for i in X.columns if i != "anemia"]
    labels = {i: LabelEncoder().fit(X[i]) for i in X.columns if X[i].dtypes == "category" or X[i].dtypes == "object"}
    scaler = RobustScaler().fit(X[numeric])
    
    X[numeric] = scaler.transform(X[numeric])
    for i in labels:
        X[i] = labels[i].transform(X[i])
    y = y.map({"ckd": 0, "notckd": 1})
    X["klasifikasi"] = y    
    # evaluation
    y_pred = models.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="notckd")
    recall = recall_score(y_test, y_pred, pos_label="notckd")
    f1 = f1_score(y_test, y_pred, pos_label="notckd")
    
    return df, models, accuracy, precision, recall, f1, X, df_new

(
    df, models, accuracy,
    precision, recall, f1, df2, df3
) = model("./ginjal/ginjal.csv", "./ginjal/penyakit_ginjal_kronik.csv")

option = st.sidebar.selectbox(
    "Silakan pilih:",
    ("Home","Dataframe", "Model Building", "Predict")
)

if option == "Home" or option == "":
    st.write("""# Chronic Kidney Desease""") #menampilkan halaman utama
    st.write()
    st.markdown("**This project can help people to predict their desease belong to ckd or not**")
    st.write("This website is about build project Chronic Kidney Desease")
    col1, col2 = st.columns(2)
    with col1:
        st.image("danitelkom.jpeg", width=200)
    with col2:
        st.write(f"""
        Name : Dhani Munir Supriyadi\n
        Birth : 27 November 2001\n
        Degree : Bachelor degree start from 2020 until 2024\n
        Lasted GPA : 3.97 from 4.00\n
        University : University of Bina Sarana Informatika\n
        Field : Information System\n
        Linkedin : https://www.linkedin.com/in/dhani-munir-supriyadi-5b5169235/ \n
        Github : https://github.com/dhanimunir \n
        Email : danimmunir1@gmail.com \n
        Phone : +62895326168339
        """)

elif option == "Dataframe":
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    st.write()
    st.markdown("**We read the data and do step of preparation data**")
    st.write(f"\nOriginal data with {df3.shape[0]} row and {df3.shape[1]} columns")
    st.write(df3)
    st.write(f"\nAfter cleaning {df.shape[0]} row and {df.shape[1]} columns")
    st.write(df)
    st.write(f"\nAfter normalization {df2.shape[0]} row and {df2.shape[1]} columns")
    st.write(df2)
    
    

elif option == "Model Building":
    st.write("""## Model Building""")
    st.write()
    st.write("""
        We build Naive Bayes models to predict this data. The details of the performance of Naive Bayes are following below.
             """)
    st.image("./output.png")
    st.write(f"""
             The evaluation of the model is
    \n
    Accuracy Score : {accuracy*100:.1f} %\n
    Precision Score : {precision*100:.1f} %\n
    Recall Score : {recall*100:.1f} %\n
    F1 Score : {f1*100:.1f} %\n
   """)

elif option == "Predict":
    st.write("""## Predict""")
    st.write()
    results = {}
    for i in df.columns[:-1]:
        if i == "anemia":
            inputt = st.text_input(i)
        else: 
            inputt = st.number_input(i)
            
        
        results[i] = inputt 
    
    
    a = st.button("Predict")
    if a:
        results = pd.DataFrame(results, index=[0])
        proba = models.predict_proba(results)
        proba = pd.DataFrame(proba, columns=["Terkena penyakit ginjal", "Tidak terkena penyakit ginjal"])
        results = models.predict(results)
        
        if results[0] == "ckd":
            st.write("Anda terkena penyakit ginjal kronik dengan probabilitas sebagai berikut.")
        else:
            st.write("Anda tidak terkena penyakit ginjal kronik dengan probabilitas sebagai berikut")
        st.write(proba)