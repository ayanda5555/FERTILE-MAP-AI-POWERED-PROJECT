import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
import os
import hashlib
import datetime
from PIL import Image

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="🌱 FERTILE MAP - AI POWERED",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'backend', 'database', 'soil_app.db')
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'models', 'soil_classifier.h5')
UPLOAD_DIR = os.path.join(BASE_DIR, 'backend', 'uploads')

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ========== DATABASE ==========
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            farm_name TEXT DEFAULT '',
            preferred_crops TEXT DEFAULT '',
            organic_preference INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            soil_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            properties TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            crop_type TEXT DEFAULT 'general',
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()


init_db()


# ========== PASSWORD HASHING ==========
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, hashed):
    return hash_password(password) == hashed


# ========== AI MODEL ==========
@st.cache_resource
def load_model():
    """Load the soil classifier model (cached so it loads only once)"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load AI model: {str(e)}")
        return None


SOIL_CLASSES = ['chalky', 'clay', 'loamy', 'peaty', 'sandy', 'silty']

SOIL_PROPERTIES = {
    "loamy": {
        "pH_range": "6.0 - 7.0",
        "drainage": "Good",
        "nutrient_retention": "High",
        "workability": "Easy",
        "water_holding": "Moderate to High",
        "color": "Dark brown",
        "texture": "Smooth, partly gritty"
    },
    "sandy": {
        "pH_range": "5.5 - 7.0",
        "drainage": "Excessive",
        "nutrient_retention": "Low",
        "workability": "Very Easy",
        "water_holding": "Low",
        "color": "Light brown/tan",
        "texture": "Gritty, coarse"
    },
    "clay": {
        "pH_range": "6.0 - 8.0",
        "drainage": "Poor",
        "nutrient_retention": "Very High",
        "workability": "Difficult when wet",
        "water_holding": "Very High",
        "color": "Red/brown/grey",
        "texture": "Sticky, smooth"
    },
    "silty": {
        "pH_range": "6.0 - 7.0",
        "drainage": "Moderate",
        "nutrient_retention": "Moderate to High",
        "workability": "Moderate",
        "water_holding": "High",
        "color": "Dark brown",
        "texture": "Silky, flour-like"
    },
    "peaty": {
        "pH_range": "3.5 - 5.5",
        "drainage": "Poor (waterlogged)",
        "nutrient_retention": "High",
        "workability": "Easy when drained",
        "water_holding": "Very High",
        "color": "Very dark/black",
        "texture": "Spongy, fibrous"
    },
    "chalky": {
        "pH_range": "7.5 - 8.5",
        "drainage": "Good to Excessive",
        "nutrient_retention": "Low",
        "workability": "Moderate",
        "water_holding": "Low to Moderate",
        "color": "Pale white/grey",
        "texture": "Stony, gritty"
    }
}


def predict_soil(image):
    """Run AI prediction on uploaded soil image"""
    model = load_model()
    if model is None:
        return None

    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = SOIL_CLASSES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    all_predictions = {
        SOIL_CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(SOIL_CLASSES))
    }

    return {
        "soil_type": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "properties": SOIL_PROPERTIES.get(predicted_class, {}),
        "all_predictions": all_predictions
    }


# ========== FERTILIZER DATABASE ==========
FERTILIZER_DATABASE = {
    "loamy": {
        "status": "Excellent soil — ideal for most crops",
        "general_fertilizers": [
            {"name": "Balanced NPK 10-10-10", "purpose": "General maintenance", "application": "200-300 kg/hectare", "frequency": "Once per growing season"},
            {"name": "Compost / Organic Matter", "purpose": "Maintain soil structure", "application": "2-3 tonnes/hectare", "frequency": "Annually"}
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea (46-0-0)", "dose": "130 kg/ha", "timing": "Split: sowing + 30 days"},
                {"name": "DAP (18-46-0)", "dose": "100 kg/ha", "timing": "At sowing"}
            ],
            "rice": [
                {"name": "Urea", "dose": "150 kg/ha", "timing": "3 splits"},
                {"name": "MOP (0-0-60)", "dose": "50 kg/ha", "timing": "At transplanting"}
            ],
            "vegetables": [
                {"name": "NPK 19-19-19", "dose": "150 kg/ha", "timing": "At transplanting"},
                {"name": "Vermicompost", "dose": "5 tonnes/ha", "timing": "Before planting"}
            ],
            "corn": [
                {"name": "Urea", "dose": "200 kg/ha", "timing": "Split application"},
                {"name": "SSP", "dose": "150 kg/ha", "timing": "At sowing"}
            ]
        },
        "organic_alternatives": ["Cow manure compost", "Green manure (legume cover crops)", "Bone meal for phosphorus", "Wood ash for potassium"],
        "improvement_tips": ["Maintain organic matter with annual composting", "Practice crop rotation", "Avoid over-tilling to preserve structure", "Test soil pH annually"]
    },
    "sandy": {
        "status": "Needs improvement — nutrients leach quickly",
        "general_fertilizers": [
            {"name": "NPK 20-10-10 (Nitrogen-heavy)", "purpose": "Compensate for nutrient leaching", "application": "250-350 kg/hectare in splits", "frequency": "Multiple small doses"},
            {"name": "Slow-release fertilizer", "purpose": "Sustained nutrient supply", "application": "As per product label", "frequency": "Every 2-3 months"}
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea", "dose": "150 kg/ha in 3 splits", "timing": "Every 3 weeks"},
                {"name": "MOP", "dose": "80 kg/ha", "timing": "At sowing"}
            ],
            "vegetables": [
                {"name": "Liquid fertilizer", "dose": "Weekly application", "timing": "Throughout growth"},
                {"name": "NPK 15-15-15", "dose": "200 kg/ha", "timing": "Frequent small doses"}
            ],
            "corn": [
                {"name": "CAN (27-0-0)", "dose": "200 kg/ha", "timing": "Split 4 times"},
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"}
            ]
        },
        "organic_alternatives": ["Heavy composting (5+ tonnes/ha)", "Mulching to retain moisture", "Peat moss or coconut coir", "Clay amendment to improve retention"],
        "improvement_tips": ["Add organic matter heavily to improve water retention", "Use mulching to reduce evaporation", "Apply fertilizers in small, frequent doses", "Consider drip irrigation", "Plant cover crops to build soil structure"]
    },
    "clay": {
        "status": "Nutrient-rich but needs structural improvement",
        "general_fertilizers": [
            {"name": "Gypsum (calcium sulfate)", "purpose": "Break up clay, improve drainage", "application": "500-1000 kg/hectare", "frequency": "Annually"},
            {"name": "NPK 10-20-10", "purpose": "Balanced nutrition", "application": "200 kg/hectare", "frequency": "Once per season"}
        ],
        "crop_specific": {
            "wheat": [
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"},
                {"name": "Potash", "dose": "60 kg/ha", "timing": "At sowing"}
            ],
            "rice": [
                {"name": "Urea", "dose": "120 kg/ha", "timing": "Split 2-3 times"},
                {"name": "Zinc sulphate", "dose": "25 kg/ha", "timing": "At transplanting"}
            ],
            "vegetables": [
                {"name": "NPK 15-15-15", "dose": "150 kg/ha", "timing": "At planting"},
                {"name": "Compost", "dose": "5 tonnes/ha", "timing": "Before planting"}
            ]
        },
        "organic_alternatives": ["Compost to improve aeration", "Straw mulch", "Cover crops (deep-rooted)", "Green manure"],
        "improvement_tips": ["Add gypsum or sand to improve drainage", "Avoid working soil when wet", "Use raised beds for vegetables", "Add organic matter to improve structure", "Consider deep-rooted cover crops"]
    },
    "silty": {
        "status": "Fertile but prone to compaction",
        "general_fertilizers": [
            {"name": "NPK 10-10-10", "purpose": "Balanced nutrition", "application": "200 kg/hectare", "frequency": "Once per season"}
        ],
        "crop_specific": {
            "wheat": [
                {"name": "Urea", "dose": "130 kg/ha", "timing": "Split application"},
                {"name": "DAP", "dose": "100 kg/ha", "timing": "At sowing"}
            ],
            "vegetables": [
                {"name": "NPK 19-19-19", "dose": "150 kg/ha", "timing": "At planting"},
                {"name": "Compost", "dose": "4 tonnes/ha", "timing": "Before planting"}
            ]
        },
        "organic_alternatives": ["Compost", "Mulching", "Cover crops", "Green manure"],
        "improvement_tips": ["Avoid compaction — minimize foot traffic", "Add organic matter regularly", "Use mulch to prevent erosion", "Practice no-till or minimal tillage"]
    },
    "peaty": {
        "status": "Very acidic — needs pH correction",
        "general_fertilizers": [
            {"name": "Agricultural lime", "purpose": "Raise pH from acidic levels", "application": "2-4 tonnes/hectare", "frequency": "As needed based on pH test"},
            {"name": "NPK 5-10-10", "purpose": "Phosphorus and potassium boost", "application": "200 kg/hectare", "frequency": "Once per season"}
        ],
        "crop_specific": {
            "vegetables": [
                {"name": "Lime + NPK", "dose": "As per pH test", "timing": "Before planting"},
                {"name": "Bone meal", "dose": "200 kg/ha", "timing": "At planting"}
            ]
        },
        "organic_alternatives": ["Limestone to raise pH", "Wood ash", "Bone meal", "Rock phosphate"],
        "improvement_tips": ["Test and correct pH — aim for 6.0-6.5", "Improve drainage with sand/gravel channels", "Great for blueberries and acid-loving plants as-is", "Add lime gradually over seasons"]
    },
    "chalky": {
        "status": "Alkaline — may cause nutrient lockout",
        "general_fertilizers": [
            {"name": "Sulfur / Iron sulfate", "purpose": "Lower pH slightly", "application": "100-200 kg/hectare", "frequency": "Annually"},
            {"name": "NPK with micronutrients", "purpose": "Prevent iron/manganese deficiency", "application": "200 kg/hectare", "frequency": "Once per season"}
        ],
        "crop_specific": {
            "vegetables": [
                {"name": "Chelated iron", "dose": "As per label", "timing": "When deficiency appears"},
                {"name": "Acidifying fertilizer", "dose": "As per label", "timing": "Regular application"}
            ]
        },
        "organic_alternatives": ["Pine needle mulch (acidifying)", "Composted oak leaves", "Sulfur chips", "Acidic compost"],
        "improvement_tips": ["Add organic matter to buffer alkalinity", "Use chelated micronutrients", "Choose alkaline-tolerant crops", "Avoid phosphorus excess — causes lockout", "Grow lavender, spinach, beets (thrive in alkaline)"]
    }
}


def get_recommendations(soil_type, crop_type="general"):
    soil_data = FERTILIZER_DATABASE.get(soil_type, {})
    result = {
        "status": soil_data.get("status", "Unknown"),
        "general_fertilizers": soil_data.get("general_fertilizers", []),
        "organic_alternatives": soil_data.get("organic_alternatives", []),
        "improvement_tips": soil_data.get("improvement_tips", []),
    }
    if crop_type != "general":
        crop_data = soil_data.get("crop_specific", {}).get(crop_type, [])
        result["crop_specific_fertilizers"] = crop_data
        result["selected_crop"] = crop_type
    else:
        result["available_crops"] = list(soil_data.get("crop_specific", {}).keys())
    return result


# ========== EDUCATION DATA ==========
SOIL_EDUCATION = {
    "loamy": {
        "title": "Loamy Soil",
        "description": "The gold standard for agriculture. Balanced mix of sand, silt, and clay.",
        "composition": {"Sand": 40, "Silt": 40, "Clay": 20},
        "best_crops": ["Wheat", "Corn", "Tomatoes", "Peppers", "Most vegetables"],
        "characteristics": ["Dark brown", "Crumbly texture", "Holds moisture", "Rich in nutrients", "Easy to work"]
    },
    "sandy": {
        "title": "Sandy Soil",
        "description": "Large particles, drains quickly, low nutrient retention.",
        "composition": {"Sand": 70, "Silt": 15, "Clay": 15},
        "best_crops": ["Carrots", "Potatoes", "Lettuce", "Strawberries", "Herbs"],
        "characteristics": ["Light brown", "Gritty texture", "Fast draining", "Warms quickly", "Low nutrients"]
    },
    "clay": {
        "title": "Clay Soil",
        "description": "Fine particles, retains water and nutrients but poor drainage.",
        "composition": {"Sand": 20, "Silt": 20, "Clay": 60},
        "best_crops": ["Rice", "Wheat", "Broccoli", "Cabbage", "Beans"],
        "characteristics": ["Red/brown/grey", "Sticky when wet", "Hard when dry", "High nutrients", "Poor drainage"]
    },
    "silty": {
        "title": "Silty Soil",
        "description": "Medium particles, fertile but prone to compaction.",
        "composition": {"Sand": 20, "Silt": 60, "Clay": 20},
        "best_crops": ["Most vegetables", "Grasses", "Shrubs"],
        "characteristics": ["Dark brown", "Silky texture", "Holds moisture", "Fertile", "Erosion prone"]
    },
    "peaty": {
        "title": "Peaty Soil",
        "description": "Rich in organic matter, very acidic, found in marshy areas.",
        "composition": {"Organic Matter": 70, "Mineral": 30},
        "best_crops": ["Blueberries", "Potatoes", "Heather"],
        "characteristics": ["Very dark/black", "Spongy", "Highly acidic", "Waterlogged", "Organic rich"]
    },
    "chalky": {
        "title": "Chalky Soil",
        "description": "Alkaline, stony, overlies limestone bedrock.",
        "composition": {"Calcium Carbonate": 40, "Sand": 30, "Clay": 30},
        "best_crops": ["Lavender", "Spinach", "Beets", "Sweet corn"],
        "characteristics": ["Pale/white", "Stony", "Very alkaline", "Free-draining", "Causes yellowing"]
    }
}

NPK_GUIDE = {
    "nitrogen": {"symbol": "N", "role": "Leaf and stem growth", "deficiency_signs": "Yellow leaves", "excess_signs": "Too much foliage"},
    "phosphorus": {"symbol": "P", "role": "Root and flower development", "deficiency_signs": "Purple leaves", "excess_signs": "Blocks zinc/iron"},
    "potassium": {"symbol": "K", "role": "Overall health and disease resistance", "deficiency_signs": "Brown leaf edges", "excess_signs": "Blocks calcium"}
}


# ========== SESSION STATE ==========
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'farm_name' not in st.session_state:
    st.session_state.farm_name = ""


# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .soil-card {
        background: #f0f7f0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)


# ========== LOGIN / REGISTER PAGE ==========
def show_auth_page():
    st.markdown('<h1 class="main-header">🌱 FERTILE MAP - AI POWERED</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Soil Fertility Mapping Application</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        st.subheader("Welcome Back!")
        with st.form("login_form"):
            email = st.text_input("📧 Email")
            password = st.text_input("🔒 Password", type="password")
            login_btn = st.form_submit_button("🔐 Login", use_container_width=True)

            if login_btn:
                if email and password:
                    db = get_db()
                    user = db.execute(
                        'SELECT * FROM users WHERE email = ?', (email,)
                    ).fetchone()
                    db.close()

                    if user and verify_password(password, user['password_hash']):
                        st.session_state.logged_in = True
                        st.session_state.user_id = user['id']
                        st.session_state.user_name = user['full_name']
                        st.session_state.user_email = user['email']
                        st.session_state.farm_name = user['farm_name']
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                else:
                    st.error("Please fill in all fields")

    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            full_name = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email Address")
            farm_name = st.text_input("🌾 Farm Name (optional)")
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            register_btn = st.form_submit_button("📝 Register", use_container_width=True)

            if register_btn:
                if not full_name or not email or not password:
                    st.error("Please fill in all required fields")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    db = get_db()
                    existing = db.execute(
                        'SELECT id FROM users WHERE email = ?', (email,)
                    ).fetchone()

                    if existing:
                        st.error("❌ Email already registered")
                    else:
                        password_hashed = hash_password(password)
                        db.execute(
                            'INSERT INTO users (email, password_hash, full_name, farm_name) VALUES (?, ?, ?, ?)',
                            (email, password_hashed, full_name, farm_name)
                        )
                        db.commit()
                        st.success("✅ Registration successful! Please login.")
                    db.close()


# ========== HOME PAGE ==========
def show_home():
    st.markdown('<h1 class="main-header">🌱 FERTILE MAP - AI POWERED</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, **{st.session_state.user_name}**! 👋")

    # Get stats
    db = get_db()
    total = db.execute(
        'SELECT COUNT(*) as count FROM analyses WHERE user_id = ?',
        (st.session_state.user_id,)
    ).fetchone()['count']

    avg_conf = db.execute(
        'SELECT AVG(confidence) as avg FROM analyses WHERE user_id = ?',
        (st.session_state.user_id,)
    ).fetchone()['avg']

    soil_dist = db.execute(
        '''SELECT soil_type, COUNT(*) as count FROM analyses 
           WHERE user_id = ? GROUP BY soil_type ORDER BY count DESC''',
        (st.session_state.user_id,)
    ).fetchall()

    recent = db.execute(
        '''SELECT soil_type, confidence, created_at FROM analyses 
           WHERE user_id = ? ORDER BY created_at DESC LIMIT 5''',
        (st.session_state.user_id,)
    ).fetchall()
    db.close()

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Analyses", total)
    with col2:
        st.metric("🎯 Avg Confidence", f"{(avg_conf or 0) * 100:.1f}%")
    with col3:
        st.metric("🌾 Farm", st.session_state.farm_name or "Not set")

    st.markdown("---")

    # Soil Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Soil Type Distribution")
        if soil_dist:
            dist_df = pd.DataFrame(
                [{"Soil Type": s['soil_type'].title(), "Count": s['count']} for s in soil_dist]
            )
            st.bar_chart(dist_df.set_index("Soil Type"))
        else:
            st.info("No analyses yet. Upload a soil image to get started!")

    with col2:
        st.subheader("🕐 Recent Analyses")
        if recent:
            for r in recent:
                st.markdown(f"""
                <div class="soil-card">
                    <strong>{r['soil_type'].title()}</strong> — 
                    Confidence: {r['confidence'] * 100:.1f}% — 
                    {r['created_at'][:10]}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent analyses.")


# ========== SOIL ANALYSIS PAGE ==========
def show_analysis():
    st.title("📸 Soil Analysis")
    st.markdown("Upload a photo of your soil for AI-powered analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Soil Image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Take a clear photo of your soil sample"
        )

        crop_type = st.selectbox(
            "🌾 Select Crop Type",
            ["general", "wheat", "rice", "corn", "vegetables"],
            help="Choose a crop for specific fertilizer recommendations"
        )

        analyze_btn = st.button("🔬 Analyze Soil", use_container_width=True)

    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Soil Image", use_container_width=True)

    if analyze_btn and uploaded_file:
        with st.spinner("🤖 AI is analyzing your soil..."):
            image = Image.open(uploaded_file)
            prediction = predict_soil(image)

            if prediction is None:
                st.error("❌ Analysis failed. AI model could not be loaded.")
                return

            recommendations = get_recommendations(prediction['soil_type'], crop_type)

            # Save image
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"user{st.session_state.user_id}_{timestamp}_{uploaded_file.name}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            image.save(filepath)

            # Save to database
            db = get_db()
            db.execute(
                '''INSERT INTO analyses 
                   (user_id, image_path, soil_type, confidence, properties, recommendations, crop_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (st.session_state.user_id, filename, prediction['soil_type'],
                 prediction['confidence'], json.dumps(prediction['properties']),
                 json.dumps(recommendations), crop_type)
            )
            db.commit()
            db.close()

        # ===== SHOW RESULTS =====
        st.markdown("---")
        st.header("📊 Analysis Results")

        # Soil Type & Confidence
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏷️ Soil Type", prediction['soil_type'].title())
        with col2:
            confidence_pct = prediction['confidence_percent']
            st.metric("🎯 Confidence", f"{confidence_pct}%")
        with col3:
            st.metric("📋 Status", recommendations['status'][:30])

        # Confidence bar
        if confidence_pct >= 80:
            st.success(f"✅ High confidence: {confidence_pct}%")
        elif confidence_pct >= 50:
            st.warning(f"⚠️ Moderate confidence: {confidence_pct}%")
        else:
            st.error(f"🔴 Low confidence: {confidence_pct}%")

        # All predictions
        st.subheader("📊 All Soil Type Probabilities")
        pred_df = pd.DataFrame(
            list(prediction['all_predictions'].items()),
            columns=['Soil Type', 'Probability (%)']
        )
        pred_df = pred_df.sort_values('Probability (%)', ascending=False)
        st.bar_chart(pred_df.set_index('Soil Type'))

        # Soil Properties
        st.subheader("🔬 Soil Properties")
        props = prediction['properties']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**pH Range:** {props.get('pH_range', 'N/A')}")
            st.markdown(f"**Color:** {props.get('color', 'N/A')}")
        with col2:
            st.markdown(f"**Drainage:** {props.get('drainage', 'N/A')}")
            st.markdown(f"**Texture:** {props.get('texture', 'N/A')}")
        with col3:
            st.markdown(f"**Nutrients:** {props.get('nutrient_retention', 'N/A')}")
            st.markdown(f"**Workability:** {props.get('workability', 'N/A')}")
        with col4:
            st.markdown(f"**Water Holding:** {props.get('water_holding', 'N/A')}")

        # Fertilizer Recommendations
        st.subheader("🧪 Fertilizer Recommendations")

        # General fertilizers
        st.markdown("#### 💊 Recommended Fertilizers")
        for fert in recommendations.get('general_fertilizers', []):
            with st.expander(f"🔹 {fert['name']}"):
                st.write(f"**Purpose:** {fert['purpose']}")
                st.write(f"**Application:** {fert['application']}")
                st.write(f"**Frequency:** {fert['frequency']}")

        # Crop-specific
        if 'crop_specific_fertilizers' in recommendations and recommendations['crop_specific_fertilizers']:
            st.markdown(f"#### 🌾 Specific for: **{crop_type.title()}**")
            for fert in recommendations['crop_specific_fertilizers']:
                with st.expander(f"🔹 {fert['name']}"):
                    st.write(f"**Dose:** {fert['dose']}")
                    st.write(f"**Timing:** {fert['timing']}")

        # Organic Alternatives
        st.markdown("#### 🌿 Organic Alternatives")
        for alt in recommendations.get('organic_alternatives', []):
            st.write(f"- 🌿 {alt}")

        # Improvement Tips
        st.markdown("#### 💡 Soil Improvement Tips")
        for tip in recommendations.get('improvement_tips', []):
            st.write(f"- 💡 {tip}")

    elif analyze_btn and not uploaded_file:
        st.error("❌ Please upload an image first!")


# ========== HISTORY PAGE ==========
def show_history():
    st.title("📋 Analysis History")

    db = get_db()
    analyses = db.execute(
        'SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC',
        (st.session_state.user_id,)
    ).fetchall()
    db.close()

    if not analyses:
        st.info("📭 No analyses yet. Go to 'Soil Analysis' to upload your first soil image!")
        return

    st.write(f"**Total analyses:** {len(analyses)}")

    for a in analyses:
        properties = json.loads(a['properties'])
        recommendations = json.loads(a['recommendations'])

        with st.expander(
            f"🏷️ {a['soil_type'].title()} — "
            f"Confidence: {a['confidence'] * 100:.1f}% — "
            f"📅 {a['created_at'][:10]}"
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                # Show image if exists
                img_path = os.path.join(UPLOAD_DIR, a['image_path'])
                if os.path.exists(img_path):
                    st.image(img_path, caption="Soil Image", use_container_width=True)
                else:
                    st.write("📷 Image not available")

            with col2:
                st.markdown(f"**Soil Type:** {a['soil_type'].title()}")
                st.markdown(f"**Confidence:** {a['confidence'] * 100:.1f}%")
                st.markdown(f"**Crop Type:** {a['crop_type'].title()}")
                st.markdown(f"**Date:** {a['created_at']}")
                st.markdown(f"**Status:** {recommendations.get('status', 'N/A')}")

                st.markdown("**Properties:**")
                for key, val in properties.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {val}")

            # Delete button
            if st.button(f"🗑️ Delete", key=f"del_{a['id']}"):
                db = get_db()
                db.execute(
                    'DELETE FROM analyses WHERE id = ? AND user_id = ?',
                    (a['id'], st.session_state.user_id)
                )
                db.commit()
                db.close()
                st.success("✅ Analysis deleted!")
                st.rerun()

    # Download all as CSV
    st.markdown("---")
    db = get_db()
    df = pd.read_sql_query(
        'SELECT soil_type, confidence, crop_type, created_at FROM analyses WHERE user_id = ?',
        db, params=(st.session_state.user_id,)
    )
    db.close()

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="fertile_map_history.csv",
        mime="text/csv",
        use_container_width=True
    )


# ========== EDUCATION PAGE ==========
def show_education():
    st.title("📚 Soil Education Center")

    tab1, tab2 = st.tabs(["🏷️ Soil Types", "🧪 Fertilizer Guide"])

    with tab1:
        st.subheader("Learn About Different Soil Types")

        for soil_key, soil_info in SOIL_EDUCATION.items():
            with st.expander(f"🔹 {soil_info['title']}"):
                st.markdown(f"**{soil_info['description']}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Composition:**")
                    comp_df = pd.DataFrame(
                        list(soil_info['composition'].items()),
                        columns=['Component', 'Percentage']
                    )
                    st.bar_chart(comp_df.set_index('Component'))

                with col2:
                    st.markdown("**Best Crops:**")
                    for crop in soil_info['best_crops']:
                        st.write(f"- 🌾 {crop}")

                st.markdown("**Characteristics:**")
                for char in soil_info['characteristics']:
                    st.write(f"- ✅ {char}")

    with tab2:
        st.subheader("Understanding N-P-K Ratios")
        st.markdown("Three numbers on fertilizer bags represent **Nitrogen (N)**, **Phosphorus (P)**, and **Potassium (K)**.")

        for nutrient, info in NPK_GUIDE.items():
            with st.expander(f"🔹 {info['symbol']} — {nutrient.title()}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Role:** {info['role']}")
                    st.markdown(f"**Deficiency Signs:** {info['deficiency_signs']}")
                with col2:
                    st.markdown(f"**Excess Signs:** {info['excess_signs']}")

        st.markdown("---")
        st.subheader("Application Methods")
        methods = [
            {"name": "Broadcasting", "desc": "Spread evenly across field", "best": "Pre-planting"},
            {"name": "Banding", "desc": "Place near seed rows", "best": "Row crops"},
            {"name": "Side-dressing", "desc": "Apply alongside plants", "best": "Mid-season boost"},
            {"name": "Foliar Spray", "desc": "Spray on leaves", "best": "Quick fix"}
        ]
        for m in methods:
            st.markdown(f"- **{m['name']}**: {m['desc']} *(Best for: {m['best']})*")


# ========== PROFILE PAGE ==========
def show_profile():
    st.title("👤 My Profile")

    db = get_db()
    user = db.execute(
        'SELECT * FROM users WHERE id = ?', (st.session_state.user_id,)
    ).fetchone()
    db.close()

    if user:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Profile Information")
            st.write(f"**👤 Name:** {user['full_name']}")
            st.write(f"**📧 Email:** {user['email']}")
            st.write(f"**🌾 Farm:** {user['farm_name'] or 'Not set'}")
            st.write(f"**📅 Joined:** {user['created_at'][:10]}")

        with col2:
            st.markdown("### ✏️ Edit Profile")
            with st.form("profile_form"):
                new_name = st.text_input("Full Name", value=user['full_name'])
                new_farm = st.text_input("Farm Name", value=user['farm_name'] or '')

                if st.form_submit_button("💾 Update Profile", use_container_width=True):
                    db = get_db()
                    db.execute(
                        'UPDATE users SET full_name=?, farm_name=? WHERE id=?',
                        (new_name, new_farm, st.session_state.user_id)
                    )
                    db.commit()
                    db.close()

                    st.session_state.user_name = new_name
                    st.session_state.farm_name = new_farm
                    st.success("✅ Profile updated!")
                    st.rerun()


# ========== SIDEBAR & NAVIGATION ==========
if not st.session_state.logged_in:
    show_auth_page()
else:
    # Sidebar
    st.sidebar.title("🌱 FERTILE MAP")
    st.sidebar.markdown(f"👤 **{st.session_state.user_name}**")
    st.sidebar.markdown(f"🌾 {st.session_state.farm_name or 'No farm set'}")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "📸 Soil Analysis", "📋 History",
         "📚 Education", "👤 Profile"]
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.farm_name = ""
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by FERTILE MAP Team")
    st.sidebar.markdown("© 2024 All Rights Reserved")

    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "📸 Soil Analysis":
        show_analysis()
    elif page == "📋 History":
        show_history()
    elif page == "📚 Education":
        show_education()
    elif page == "👤 Profile":
        show_profile()
