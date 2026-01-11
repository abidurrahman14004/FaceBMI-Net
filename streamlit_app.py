"""
This file allows Streamlit Cloud to recognize the project.
For Flask apps, consider using Render, Railway, or Heroku instead.
"""
import os

# If running on Streamlit Cloud, show a message
if os.environ.get('STREAMLIT_SERVER_PORT'):
    import streamlit as st
    st.set_page_config(page_title="BMI Predictor", page_icon="🤖")
    st.error("""
    ⚠️ **This is a Flask application, not a Streamlit app.**
    
    Streamlit Cloud is designed for Streamlit apps. For Flask apps, please use:
    - **Render.com** (recommended) - Free tier available
    - **Railway.app** - Easy deployment
    - **Heroku** - Classic platform
    - **Fly.io** - Modern alternative
    
    Your Flask app works perfectly on localhost. To deploy:
    1. Push to GitHub
    2. Connect to Render/Railway
    3. Set build command: `pip install -r requirements.txt`
    4. Set start command: `gunicorn app:app`
    """)
else:
    # For local or other platforms, just run Flask
    from app import app
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
