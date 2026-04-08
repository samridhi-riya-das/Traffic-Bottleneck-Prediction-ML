# Traffic-Bottleneck-ML
##  Team

Team Name: SYNAPSE

- Samridhi Riya Das  
- Jai Soni  

---

##  Institution
Shri Shankaracharya Technical Campus (SSTC)

##  Workflow

1. Data Collection  
   Traffic dataset with speed, density, time, weather, and road type is used.

2. Data Preprocessing  
   - Traffic density levels converted into numerical values  
   - Data cleaned and structured for ML  

3. Feature Engineering  
   - Lane Deviation Index (LDI) → measures disorder in traffic  
   - Merge Conflict Rate (MCR) → measures merging conflicts  

4. Model Training  
   - Random Forest classifier trained on engineered features  

5. Prediction  
   - Model predicts traffic level (Low / Medium / High)  

6. Visualization  
   - Traffic distribution  
   - Speed vs Density  
   - Feature importance
  
## 🔮 Future Scope

Our current system is a prototype, but it can be extended into a real-world smart traffic solution:

### 🌐 IoT Sensor Integration
- Connect with vehicles 
- Continuous data flow for live prediction  

### 🗺️ Live Maps Integration
- Integrate with mapping systems (Google Maps, etc.)  
- Visualize congestion zones dynamically  

### 🛣️ Live Road Condition Monitoring
- Include road quality, accidents, and weather impacts  
- Improve prediction accuracy  

### 🚦 Smart Traffic Signal Optimization
- Use predictions to control traffic signals dynamically  
- Reduce congestion at bottlenecks  

### 🧭 Intelligent Route Navigation
- Suggest alternative routes based on congestion prediction  
- Improve travel time and efficiency

## 🚀 How to Run our demo

1. Download or clone the repository
2. Open terminal in the project folder

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

7. Deployment  
   - Streamlit app for real-time user input and prediction
  
## 💻 Running in VS Code

1. Open the project folder in VS Code  
2. Open terminal (Ctrl + `)  
3. Install dependencies:
   pip install -r requirements.txt  
4. Run the app:
   streamlit run app.py  
5. The app will open in your browser  
