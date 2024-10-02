import streamlit as st
import subprocess
import time
import requests
import os

# Set the page configuration
st.set_page_config(page_title="IntelliSeason", page_icon=":sun_with_face:", layout="centered")

# Custom CSS for styling, including dark mode support
st.markdown("""
    <style>
    /* Light mode styles */
    body {
        background-color: #f0f4ff; /* Light background color */
        font-family: 'Arial', sans-serif; /* Clean font */
        color: black; /* Set font color to black */
    }
    .center-container {
        display: flex;
        justify-content: center; 
        align-items: center; 
        flex-direction: column; 
        margin-top: 20px;
    }
    .stButton > button {
        color: white;
        background-color: #2D13EA; /* Updated blue color */
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 18px; /* Increased font size */
        margin: 10px;
        cursor: pointer;
        border-radius: 8px; /* Softer corners */
        transition: background-color 0.3s ease, transform 0.2s; /* Add transform on hover */
        width: 220px; 
    }
    .stButton > button:hover {
        background-color: #1A0DB8; /* Darker blue on hover */
        transform: scale(1.05); /* Slightly enlarge button */
    }
    h1, h2, h3 {
        color: black;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: grey;
    }
    .highlight {
        color: #E94E77; /* Highlight color for emphasis */
        font-weight: bold;
    }
    .instructions {
        background-color: #ffffff; /* White background for instructions */
        border-radius: 8px; /* Rounded corners */
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #1c1c1c; /* Dark background color */
            color: #f0f4ff; /* Light font color */
        }
        .stButton > button {
            color: white;
            background-color: #5b34fa; /* Light purple color */
        }
        .stButton > button:hover {
            background-color: #3c22d1; /* Darker purple on hover */
        }
        h1, h2, h3 {
            color: #f0f4ff; /* Light header color */
        }
        .footer {
            color: #a9a9a9; /* Grey footer in dark mode */
        }
        .instructions {
            background-color: #333333; /* Dark background for instructions */
            box-shadow: 0 2px 10px rgba(255, 255, 255, 0.1); /* Light shadow in dark mode */
        }
    }
    </style>
""", unsafe_allow_html=True)

# Omantel logo
st.image("logo.png", width=200)  # Replace with the correct path to your uploaded logo

# Title and subtitle
st.title("Welcome to IntelliSeason!")
st.subheader("Your seasonal insights, powered by Omantel")

# Description
st.write("""
If you’re new to our platform, please sign up to unlock a wealth of seasonal insights. 
If you already have an account, simply log in to continue your journey with us.
""")

# Function to check if the Flask server is running on port 5000
def is_flask_server_running(url="http://localhost:5000"):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# Function to start the Flask server
def start_flask_server():
    # Run flask_server2.py in a separate process on port 5000
    subprocess.Popen(["python3.11", "flask_server_2.py"])

# Center the buttons using a div container
st.markdown('<div class="center-container">', unsafe_allow_html=True)

# Create the buttons with immediate action
col1, col2 = st.columns(2)

with col1:
    if st.button("Login"):
        # Check if the Flask server is running on port 5000
        if not is_flask_server_running():
            st.write("Flask server not running. Starting the server...")
            start_flask_server()
            time.sleep(5)  # Give the server a few seconds to start
            
        # Redirect to Flask server on localhost:5000
        login_url = "http://localhost:5000"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={login_url}">', unsafe_allow_html=True)

with col2:
    if st.button("Signup"):
        # Redirect immediately to the Google Form
        signup_url = "https://docs.google.com/forms/d/e/1FAIpQLSfxrWnmynMWlUZNYdguiHViOV7Ca8yImWQbCQDHqgBgQYsDrQ/viewform"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={signup_url}">', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Divider
st.write("---")

# User guide section
st.write("### Important: Create a Gmail Account Before Signing Up")
st.write("Before you can sign up for IntelliSeason, you need to have a Gmail account using your Omantel ID. If you don’t have one, please follow these steps:")

st.write("### Sign up for a Gmail account using your Omantel ID:")
st.write("1. Go to the [Google Account sign-in page](https://accounts.google.com/).")
st.write("2. Click on **'Create account'** and select **'For myself.'**")
st.write("3. Choose the option to **'Use your existing email.'**")
st.write("4. Follow the prompts to complete the account creation process.")

st.write("### Sign in to IntelliSeason using your Gmail account:")
st.write("1. Go to the IntelliSeason sign-in page.")
st.write("2. Select the option to **'Signup.'**")

# Footer
st.markdown('<div class="footer">© 2024 Omantel. All rights reserved.</div>', unsafe_allow_html=True)

# Main function - run Streamlit on port 8502 (manually or externally handled)
if __name__ == "__main__":
    # Start Streamlit on port 8502 automatically without using subprocess
    pass  # No subprocess call to avoid conflict
