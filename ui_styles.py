import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f3a93, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 1rem;
    }
    
    .sidebar-section {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1976d2;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #37474f 0%, #546e7a 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .detection-stats {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    
    .template-box {
        background: #f0f4ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    
    /* Consistent image sizing for drone gallery */
    .stImage > img {
        max-height: 200px;
        object-fit: contain;
        width: 100% !important;
    }
    
    /* Target the Unidentified Drones tab gallery - force 100px containers */
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] {
        min-height: 300px;
        max-height: 300px;
        padding: 10px;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        margin: 5px;
        display: flex;
        flex-direction: column;
        align-items: center;
        background-color: #fafafa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Force ALL images in drone gallery to 100x100px */
    div[data-testid="tabpanel"]:nth-child(4) .stImage {
        width: 100px !important;
        height: 100px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        margin: 0 auto 10px auto !important;
        border: 2px solid #ccc;
        border-radius: 8px;
        flex-shrink: 0 !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) .stImage > div {
        width: 100px !important;
        height: 100px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) .stImage img {
        max-width: 100px !important;
        max-height: 100px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 4px;
        display: block !important;
    }
    
    /* Ensure all images in the gallery maintain proper proportions */
    div[data-testid="tabpanel"]:nth-child(4) img {
        max-width: 100px !important;
        max-height: 100px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        object-position: center !important;
    }
    
    /* Force consistent button sizing in gallery */
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] .stButton {
        margin: 2px 0 !important;
    }
    
    div[data-testid="tabpanel"]:nth-child(4) div[data-testid="column"] .stButton button {
        font-size: 12px !important;
        padding: 4px 8px !important;
        height: 32px !important;
    }
    
    /* Template images consistent sizing */
    .template-image {
        max-height: 150px;
        object-fit: contain;
    }
    
    /* Detection result images */
    .detection-image {
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
</style>""", unsafe_allow_html=True)
