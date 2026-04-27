import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="🌱 FERTILE MAP - AI POWERED",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== DATABASE CONNECTION ==========
def get_connection():
    conn = sqlite3.connect('fertile_map.db', check_same_thread=False)
    return conn

# ========== CREATE TABLES ==========
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS soil_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            ph_level REAL,
            moisture REAL,
            fertility_score REAL,
            date_recorded TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_tables()

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.title("🌱 FERTILE MAP")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🗺️ Fertility Map", "📊 Analysis", 
     "🤖 AI Prediction", "📝 Add Data", "📋 View Data"]
)

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.title("🌱 FERTILE MAP - AI POWERED")
    st.markdown("### AI-Powered Soil Fertility Mapping Application")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📍 Locations Mapped", value="150+")
    with col2:
        st.metric(label="🧪 Soil Samples", value="500+")
    with col3:
        st.metric(label="🎯 Accuracy", value="95%")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 Features
    - 🗺️ Interactive soil fertility mapping
    - 🤖 AI-powered soil analysis
    - 📊 Data visualization and reports
    - 📱 Works on mobile and desktop
    """)

# ========== FERTILITY MAP PAGE ==========
elif page == "🗺️ Fertility Map":
    st.title("🗺️ Soil Fertility Map")
    
    # Sample map data
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM soil_data", conn)
    conn.close()
    
    if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
        st.map(df[['latitude', 'longitude']])
    else:
        # Default sample data
        map_data = pd.DataFrame({
            'latitude': [12.97, 12.95, 12.99, 13.01, 12.93],
            'longitude': [77.59, 77.57, 77.61, 77.55, 77.63]
        })
        st.map(map_data)
    
    st.info("📍 Each point represents a soil sample location")

# ========== ANALYSIS PAGE ==========
elif page == "📊 Analysis":
    st.title("📊 Soil Analysis Dashboard")
    
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM soil_data", conn)
    conn.close()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Nitrogen Levels")
            st.bar_chart(df[['location', 'nitrogen']].set_index('location'))
        
        with col2:
            st.subheader("pH Levels")
            st.bar_chart(df[['location', 'ph_level']].set_index('location'))
        
        st.subheader("Complete Data")
        st.dataframe(df)
    else:
        st.warning("No data available. Please add soil data first.")

# ========== AI PREDICTION PAGE ==========
elif page == "🤖 AI Prediction":
    st.title("🤖 AI Fertility Prediction")
    st.markdown("Enter soil parameters to predict fertility:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nitrogen = st.slider("Nitrogen (N)", 0.0, 100.0, 50.0)
        phosphorus = st.slider("Phosphorus (P)", 0.0, 100.0, 30.0)
        potassium = st.slider("Potassium (K)", 0.0, 100.0, 25.0)
    
    with col2:
        ph_level = st.slider("pH Level", 0.0, 14.0, 6.5)
        moisture = st.slider("Moisture (%)", 0.0, 100.0, 40.0)
    
    if st.button("🔮 Predict Fertility", use_container_width=True):
        # Simple prediction logic (replace with your AI model)
        score = (nitrogen * 0.3 + phosphorus * 0.2 + 
                potassium * 0.2 + (14 - abs(ph_level - 6.5)) * 2 + 
                moisture * 0.1)
        score = min(max(score, 0), 100)
        
        st.markdown("---")
        
        if score >= 80:
            st.success(f"🌟 Fertility Score: {score:.1f}/100 - HIGHLY FERTILE")
        elif score >= 50:
            st.warning(f"⚠️ Fertility Score: {score:.1f}/100 - MODERATELY FERTILE")
        else:
            st.error(f"🔴 Fertility Score: {score:.1f}/100 - LOW FERTILITY")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if nitrogen < 30:
            st.write("- 🔹 Add nitrogen-rich fertilizers")
        if phosphorus < 20:
            st.write("- 🔹 Add phosphorus supplements")
        if potassium < 20:
            st.write("- 🔹 Add potassium-based fertilizers")
        if ph_level < 5.5 or ph_level > 7.5:
            st.write("- 🔹 Adjust soil pH level")

# ========== ADD DATA PAGE ==========
elif page == "📝 Add Data":
    st.title("📝 Add Soil Data")
    
    with st.form("soil_form"):
        location = st.text_input("📍 Location Name")
        
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=0.0, format="%.6f")
            nitrogen = st.number_input("Nitrogen (N)", value=0.0)
            phosphorus = st.number_input("Phosphorus (P)", value=0.0)
        with col2:
            longitude = st.number_input("Longitude", value=0.0, format="%.6f")
            potassium = st.number_input("Potassium (K)", value=0.0)
            ph_level = st.number_input("pH Level", value=0.0)
        
        moisture = st.number_input("Moisture (%)", value=0.0)
        
        submitted = st.form_submit_button("💾 Save Data", use_container_width=True)
        
        if submitted:
            if location:
                conn = get_connection()
                cursor = conn.cursor()
                
                # Calculate fertility score
                score = (nitrogen * 0.3 + phosphorus * 0.2 + 
                        potassium * 0.2 + moisture * 0.1)
                
                cursor.execute('''
                    INSERT INTO soil_data 
                    (location, latitude, longitude, nitrogen, phosphorus, 
                     potassium, ph_level, moisture, fertility_score, date_recorded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (location, latitude, longitude, nitrogen, phosphorus,
                      potassium, ph_level, moisture, score))
                
                conn.commit()
                conn.close()
                st.success("✅ Data saved successfully!")
            else:
                st.error("Please enter a location name")

# ========== VIEW DATA PAGE ==========
elif page == "📋 View Data":
    st.title("📋 All Soil Data")
    
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM soil_data", conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="fertile_map_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available yet.")

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by FERTILE MAP Team")
st.sidebar.markdown("© 2024 All Rights Reserved")