import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# make container
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Data apps for ship")
    st.text("This app is created to predict the ship type")

with dataset:
    st.header("Dataset")
    st.text("This is the ship dataset")
    # load data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.dataframe(df.head(10))
    
    # show data
    st.write(df.head(10))
    # plot a graph
    st.bar_chart(df['sex'].value_counts())

    # other plot
    st.bar_chart(df['age'].sample(10))

with features:
    st.header("Features")
    st.text("This is the ship features")
    # markdown 
    st.markdown("1. **Features 1:** This is a Titanic Graph")
with model_training:
    st.header("Model Training")
    st.text("This is the ship model training")
    # making column
    input, display = st.columns(2)

    # First column selection points
    max_depth=input.slider("How many people age do you know", min_value=10, max_value=100, value=20, step=5)

# n_estimators
n_estimators = input.selectbox("How many tree should be there in a RF?", options=[50,100,150,200,250,300, 'No Limit'], index=0)


# adding list of features
input.write(df.columns)

# input feature from user
input_features = input.text_input("Which features we should use?")

# display the result
display.write("You have selected {} as n_estimators".format(n_estimators))
display.write("You have selected {} as features".format(input_features))


# Machine Learning Model

model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators, random_state=0)

# Condition
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth, random_state=0)
else:
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators, random_state=0)

# split data

X = df[[input_features]]
y = df[['fare']]

# fit the model
model.fit(X,y)

# predict
y_pred = model.predict(X)


# Dispaly metrices
display.subheader("Mean absolute error of the model is")  
display.write(mean_absolute_error(y, y_pred))
display.subheader("Mean squared error of the model is")
display.write(mean_squared_error(y, y_pred))
display.subheader("R2 score of the model is")
display.write(r2_score(y, y_pred)) 