import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration
st.set_page_config(
    page_title="Date Variety Intelligence Platform",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Glassmorphic CSS with Background Image
custom_css = """
<style>
/* Background Image with Overlay */
.stApp {
    background: linear-gradient(rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.88)),
                url('https://images.unsplash.com/photo-1546548970-71785318a17b?auto=format&fit=crop&w=2000&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #F8FAFC;
    font-family: 'Inter', sans-serif;
}

/* Containers & Cards styling for max visibility */
.glass-card {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 24px;
}

/* Visibility enhancements for headers */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.8);
}

.main-title {
    font-size: 2.8rem;
    background: linear-gradient(90deg, #F59E0B, #FBBF24, #FDE68A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 8px;
    font-weight: 800 !important;
}

.sub-title {
    text-align: center;
    color: #E2E8F0 !important;
    font-size: 1.1rem;
    margin-bottom: 28px;
}

/* Inputs High Contrast Styling */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #F59E0B !important;
    border-radius: 8px !important;
}

.stSlider > div {
    color: #FFFFFF !important;
}

/* Labels visibility */
label {
    color: #F8FAFC !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.75);
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #CBD5E1;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #F59E0B !important;
    color: #0F172A !important;
    font-weight: 700;
}

/* Custom Buttons */
.stButton > button {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 12px 28px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px 0 rgba(245, 158, 11, 0.39);
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(245, 158, 11, 0.55);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Variety Metadata & Descriptions
CLASSES = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'SAFAVI', 'SOGAY', 'ROTANA']

VARIETY_INFO = {
    'BERHI': {
        'origin': 'Iraq / Middle East',
        'type': 'Soft & Sweet',
        'desc': 'Small to medium spherical date with golden-yellow to amber color. Highly valued for soft texture.',
        'grading': 'Premium Export Quality'
    },
    'DEGLET': {
        'origin': 'North Africa',
        'type': 'Semi-Dry',
        'desc': 'Elongated, translucent amber date known as the "Mother of all dates" with honey-like flavor.',
        'grading': 'Commercial Standard'
    },
    'DOKOL': {
        'origin': 'Commercial Regions',
        'type': 'Dry / Firm',
        'desc': 'Distinct firm texture with medium dimensions. Primarily used in commercial processing.',
        'grading': 'Processing Grade'
    },
    'IRAQI': {
        'origin': 'Mesopotamia',
        'type': 'Soft',
        'desc': 'Broad oval shape with rich caramel tone and distinct shape factor symmetry.',
        'grading': 'Regional Selection'
    },
    'SAFAVI': {
        'origin': 'Saudi Arabia (Madinah)',
        'type': 'Semi-Dry',
        'desc': 'Dark black-brown elongated date loaded with vitamins. Highly prized in international trade.',
        'grading': 'Luxury Class A'
    },
    'SOGAY': {
        'origin': 'Arabian Peninsula',
        'type': 'Two-Tone Semi-Dry',
        'desc': 'Unique date variety with a light golden base and darker brownish top.',
        'grading': 'Gourmet Variety'
    },
    'ROTANA': {
        'origin': 'Saudi Arabia',
        'type': 'Soft Fresh',
        'desc': 'Juicy, soft, and thick-bodied date with high moisture content and distinct perimeter metrics.',
        'grading': 'Local Fresh Market'
    }
}

# Feature Names list
FEATURE_NAMES = [
    'AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY', 
    'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO', 
    'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2', 
    'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB', 
    'StdRR', 'StdRG', 'StdRB', 'SkewRR', 'SkewRG', 'SkewRB', 
    'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG', 
    'EntropyRB', 'ALLRED_Mean', 'ALLGREEN_Mean', 'ALLBLUE_Mean'
]

# Model Loader with Fallback mechanism
@st.cache_resource
def load_date_model():
    model_path = 'date_fruit_classification_model.keras'
    h5_path = 'date_fruit_classification_model.h5'
    
    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model, "keras"
        elif os.path.exists(h5_path):
            model = tf.keras.models.load_model(h5_path)
            return model, "h5"
    except Exception as e:
        pass
    
    return None, "fallback"

model, model_status = load_date_model()

# Mock Dataset Generator for Visual Analytics
@st.cache_data
def get_analytical_data():
    np.random.seed(42)
    n_samples = 400
    
    classes_choice = np.random.choice(CLASSES, n_samples)
    area = []
    perimeter = []
    red_mean = []
    green_mean = []
    shape_f1 = []

    for c in classes_choice:
        if c == 'BERHI':
            area.append(np.random.normal(250000, 30000))
            perimeter.append(np.random.normal(1900, 150))
            red_mean.append(np.random.normal(125, 10))
            green_mean.append(np.random.normal(110, 8))
            shape_f1.append(np.random.normal(0.006, 0.0005))
        elif c == 'SAFAVI':
            area.append(np.random.normal(450000, 40000))
            perimeter.append(np.random.normal(2600, 200))
            red_mean.append(np.random.normal(85, 12))
            green_mean.append(np.random.normal(70, 10))
            shape_f1.append(np.random.normal(0.004, 0.0004))
        elif c == 'SOGAY':
            area.append(np.random.normal(380000, 35000))
            perimeter.append(np.random.normal(2300, 180))
            red_mean.append(np.random.normal(140, 15))
            green_mean.append(np.random.normal(120, 12))
            shape_f1.append(np.random.normal(0.005, 0.0005))
        else:
            area.append(np.random.normal(320000, 45000))
            perimeter.append(np.random.normal(2100, 220))
            red_mean.append(np.random.normal(110, 18))
            green_mean.append(np.random.normal(95, 15))
            shape_f1.append(np.random.normal(0.0052, 0.0006))
            
    df_sim = pd.DataFrame({
        'AREA': area,
        'PERIMETER': perimeter,
        'MeanRR': red_mean,
        'MeanRG': green_mean,
        'SHAPEFACTOR_1': shape_f1,
        'Class': classes_choice
    })
    return df_sim

df_analytics = get_analytical_data()

# Model Prediction Helper
def predict_sample(features_array):
    if model is not None and model_status != "fallback":
        # Scale approximation / direct model inference
        preds = model.predict(features_array)
        return preds[0]
    else:
        # Smart Heuristic Fallback Prediction Engine
        area = features_array[0][0]
        mean_rr = features_array[0][16]
        
        logits = np.zeros(len(CLASSES))
        if area < 280000:
            logits[0] += 3.5  # BERHI
            logits[2] += 2.0  # DOKOL
        elif area > 400000:
            logits[4] += 3.8  # SAFAVI
            logits[5] += 2.5  # SOGAY
        else:
            logits[1] += 2.5  # DEGLET
            logits[3] += 2.2  # IRAQI
            logits[6] += 2.4  # ROTANA
            
        if mean_rr > 130:
            logits[5] += 1.8  # SOGAY
            logits[6] += 1.2
        elif mean_rr < 90:
            logits[4] += 2.0  # SAFAVI
            
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

# Header Section
st.markdown('<div class="main-title">🌴 Date Variety Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Class Deep Learning Classification, Visual Analytics & Quality Inspection</div>', unsafe_allow_html=True)

if model_status == "fallback":
    st.info("💡 **Demo Mode Active:** Running with built-in heuristic prediction engine. Connect your trained `.keras` model file for direct ANN inference.")

# App Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Single Date Predictor", 
    "📊 Model Analytics & Insights", 
    "📜 Variety Catalog & Business Specs"
])

# TAB 1: SINGLE PREDICTOR
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Input Physical & Optical Measurement Attributes")
    st.write("Adjust the morphological dimensions and RGB color intensities to evaluate date variety classification.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📏 Morphological Dimensions")
        area = st.slider("AREA (px²)", min_value=100000, max_value=600000, value=320000, step=5000)
        perimeter = st.slider("PERIMETER (px)", min_value=1000, max_value=3500, value=2100, step=25)
        major_axis = st.slider("MAJOR_AXIS Length", min_value=300.0, max_value=900.0, value=580.0, step=5.0)
        minor_axis = st.slider("MINOR_AXIS Length", min_value=200.0, max_value=600.0, value=380.0, step=5.0)
        
    with col2:
        st.markdown("#### 📐 Shape & Geometry Factors")
        eccentricity = st.slider("ECCENTRICITY", min_value=0.3, max_value=0.99, value=0.75, step=0.01)
        roundness = st.slider("ROUNDNESS", min_value=0.4, max_value=0.98, value=0.78, step=0.01)
        compactness = st.slider("COMPACTNESS", min_value=0.5, max_value=0.95, value=0.72, step=0.01)
        shape_factor_1 = st.slider("SHAPEFACTOR_1", min_value=0.001, max_value=0.015, value=0.005, step=0.0005)

    with col3:
        st.markdown("#### 🎨 Color & Intensity Metrics")
        mean_rr = st.slider("Mean Red (MeanRR)", min_value=50.0, max_value=200.0, value=115.0, step=1.0)
        mean_rg = st.slider("Mean Green (MeanRG)", min_value=40.0, max_value=180.0, value=98.0, step=1.0)
        mean_rb = st.slider("Mean Blue (MeanRB)", min_value=30.0, max_value=160.0, value=85.0, step=1.0)
        std_rr = st.slider("Standard Dev Red (StdRR)", min_value=5.0, max_value=50.0, value=22.0, step=0.5)

    predict_btn = st.button("⚡ Classify Date Variety")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_btn or 'has_predicted' not in st.session_state:
        st.session_state['has_predicted'] = True
        
        # Prepare full feature vector (34 features)
        input_vector = np.zeros((1, 34))
        input_vector[0, 0] = area
        input_vector[0, 1] = perimeter
        input_vector[0, 2] = major_axis
        input_vector[0, 3] = minor_axis
        input_vector[0, 4] = eccentricity
        input_vector[0, 10] = roundness
        input_vector[0, 11] = compactness
        input_vector[0, 12] = shape_factor_1
        input_vector[0, 16] = mean_rr
        input_vector[0, 17] = mean_rg
        input_vector[0, 18] = mean_rb
        input_vector[0, 19] = std_rr
        
        probabilities = predict_sample(input_vector)
        top_class_idx = np.argmax(probabilities)
        top_class_name = CLASSES[top_class_idx]
        top_confidence = probabilities[top_class_idx] * 100

        res_col1, res_col2 = st.columns([1, 1.3])
        
        with res_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Prediction Result")
            st.markdown(f"<h2 style='color: #F59E0B; text-align: center;'>{top_class_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; color: #10B981;'>Confidence: {top_confidence:.2f}%</h4>", unsafe_allow_html=True)
            
            info = VARIETY_INFO[top_class_name]
            st.write(f"**Origin:** {info['origin']}")
            st.write(f"**Category:** {info['type']}")
            st.write(f"**Commercial Grade:** {info['grading']}")
            st.write(f"**Characteristics:** {info['desc']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📈 Class Probability Breakdown")
            
            prob_df = pd.DataFrame({
                'Variety': CLASSES,
                'Probability (%)': probabilities * 100
            }).sort_values('Probability (%)', ascending=True)
            
            fig = px.bar(
                prob_df, 
                x='Probability (%)', 
                y='Variety', 
                orientation='h',
                color='Probability (%)',
                color_continuous_scale='YlOrRd',
                text_auto='.1f'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF', size=12),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: MODEL ANALYTICS & INSIGHTS
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Multi-Dimensional Feature Analytics")
    st.write("Explore feature clusters and separation boundaries across varieties.")
    
    an_col1, an_col2 = st.columns(2)
    
    with an_col1:
        st.markdown("#### Morphological Grading: Area vs. Perimeter")
        fig_scatter = px.scatter(
            df_analytics,
            x='AREA',
            y='PERIMETER',
            color='Class',
            size='AREA',
            hover_data=['SHAPEFACTOR_1'],
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with an_col2:
        st.markdown("#### Color Distribution: Mean Red (RR) vs. Mean Green (RG)")
        fig_color = px.scatter(
            df_analytics,
            x='MeanRR',
            y='MeanRG',
            color='Class',
            symbol='Class',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_color.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig_color, use_container_width=True)

    st.markdown("#### Geometric Uniformity: Shape Factor Distribution")
    fig_box = px.box(
        df_analytics,
        x='Class',
        y='SHAPEFACTOR_1',
        color='Class',
        points="all"
    )
    fig_box.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color='#FFFFFF'),
        height=400
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: VARIETY CATALOG & BUSINESS USE CASES
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📜 Date Varieties Commercial Catalog & Specifications")
    st.write("Industry specifications and grading metrics across target classification varieties.")
    
    cat_cols = st.columns(3)
    for idx, (v_name, v_data) in enumerate(VARIETY_INFO.items()):
        col_target = cat_cols[idx % 3]
        with col_target:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #F59E0B; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                <h3 style="color: #F59E0B; margin-top: 0;">🌴 {v_name}</h3>
                <p><strong>Origin:</strong> {v_data['origin']}</p>
                <p><strong>Category:</strong> {v_data['type']}</p>
                <p><strong>Commercial Grade:</strong> <span style="color: #10B981; font-weight: bold;">{v_data['grading']}</span></p>
                <p style="color: #CBD5E1; font-size: 0.9rem;">{v_data['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #94A3B8; padding: 20px;">
    Date Fruit Multi-Class Neural Network Platform | Developed for Portfolio & Industrial Inspection Applications
</div>
""", unsafe_allow_html=True)