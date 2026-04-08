# Import libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Title
st.title("Traffic Bottleneck Predictor")

# Load dataset
data = pd.read_csv("delhi_traffic_features.csv")

# Rename columns
data.rename(columns={
    'average_speed_kmph': 'speed',
    'traffic_density_level': 'density_label'
}, inplace=True)

# Convert density label to numeric
data['density'] = data['density_label'].map({
    'Low': 100,
    'Medium': 250,
    'High': 400
})

# Create LDI
max_speed = data['speed'].max()
data['LDI'] = 1 - (data['speed'] / max_speed)

# Create MCR
data['MCR'] = data['density'] * data['LDI']

# Create target
def traffic_level(mcr):
    if mcr > 300:
        return "High"
    elif mcr > 150:
        return "Medium"
    else:
        return "Low"

data['traffic_level'] = data['MCR'].apply(traffic_level)

# Prepare ML model
X = data[['density', 'speed', 'LDI', 'MCR']]
y = data['traffic_level']

model = RandomForestClassifier()
model.fit(X, y)

# User input
st.subheader("Enter Traffic Conditions")

density = st.slider("Density", 50, 500, 200)
speed = st.slider("Speed (kmph)", 0, 100, 40)

# Predict button
if st.button("Predict Traffic"):
    LDI = 1 - (speed / max_speed)
    MCR = density * LDI

    prediction = model.predict([[density, speed, LDI, MCR]])

    st.write("### Results")
    st.write("LDI:", round(LDI, 2))
    st.write("MCR:", round(MCR, 2))
    st.write("Traffic Level:", prediction[0])

    # ------------------- ANALYSIS SECTION -------------------

st.subheader("📊 Traffic Insights Dashboard")

import matplotlib.pyplot as plt

# 1. Traffic Distribution (Pie Chart - more attractive)
st.write("### Traffic Distribution")
traffic_counts = data['traffic_level'].value_counts()

fig1 = plt.figure()
plt.pie(traffic_counts, labels=traffic_counts.index, autopct='%1.1f%%')
plt.title("Traffic Level Share")
st.pyplot(fig1)


# 2. Speed vs Density (colored scatter)
st.write("### Speed vs Density Relationship")

colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
color_values = data['traffic_level'].map(colors)

fig2 = plt.figure()
plt.scatter(data['density'], data['speed'], c=color_values, alpha=0.6)
plt.xlabel("Density")
plt.ylabel("Speed")
plt.title("Speed vs Density (colored by traffic level)")
st.pyplot(fig2)


# 3. Average Speed by Traffic Level (Bar Chart)
st.write("### Average Speed by Traffic Level")

avg_speed = data.groupby('traffic_level')['speed'].mean()

fig3 = plt.figure()
avg_speed.plot(kind='bar', color=['green', 'orange', 'red'])
plt.ylabel("Average Speed")
plt.title("Speed decreases as congestion increases")
st.pyplot(fig3)


# 4. Feature Importance (clean horizontal chart)
st.write("### Feature Importance")

importance = pd.Series(model.feature_importances_, index=X.columns)

fig4 = plt.figure()
importance.sort_values().plot(kind='barh')
plt.title("Which factors influence traffic most?")
st.pyplot(fig4)
fig3 = plt.figure()
importance.sort_values().plot(kind='barh')
plt.xlabel("Importance")
st.pyplot(fig3)
