import os

# Disable telemetry
os.environ["ST_DISABLE_TELEMETRY"] = "1"

import streamlit as st
import pandas as pd
from streamlit.components.v1 import iframe
from streamlit_option_menu import option_menu
from flask_server import app
import matplotlib.pyplot as plt
import re
import requests
from utils.models import run_arima_model, run_arima_plus_model, run_times_fm, run_auto_ml
from utils.models import endpoint_predict, deploy_model, delete_endpoint, run_prophet_training

# API URLs
LIST_MODELS_URL = f'{os.environ.get("FLASK_SERVER_ADDR", "http://localhost:5000")}/list_models'
LIST_ENDPOINTS_URL = f'{os.environ.get("FLASK_SERVER_ADDR", "http://localhost:5000")}/list_user_endpoints'

BATCH_SIZE_LIMIT = 1.4 * 1024 * 1024  # 1.4MB batch size limit


# Main Model Selection and Execution
def main():
    # st.title("IntelliSeason")

    # Access query parameters
    query_params = st.query_params
    # Check if 'sid' exists in the query parameters
    if 'sid' in query_params:
        sid = query_params['sid']
        st.write(f"{sid}")

    with st.sidebar:
        st.header("Options Menu")

        selected = option_menu(
            'IntelliSeason', ["Model Library", "History"], 
            icons=['search', 'info-circle'], menu_icon='intersect', default_index=0
        )


    if selected == "Model Library":
        st.title("Forecasting Model Library")

        # New subheader and download button for User Guide
        with open('IntelliSeason_New_Guide.pdf', 'rb') as f:
            st.download_button(
                label="Download User Guide",
                data=f,
                file_name="user_guide.pdf",
                mime="application/pdf"
            )


        models = {
            "AutoML": """AutoML from Vertex AI is great for working with various types of customer and network data. 
                        \nAutoML helps you build models to predict customer behavior, classify network issues, or forecast service demands without needing deep technical skills.
                        \n**Runtime**: The model takes approximately 2 hours and 15 minutes to run.""",
            "Prophet": """Prophet from Facebook is best for forecasting trends with seasonal patterns, like customer usage or call volume. 
                        \nProphet works well for predicting patterns that repeat over time, such as monthly data usage spikes or daily call traffic trends.
                        \n**Runtime**: 1 hour 30 minutes""",
            "ARIMA": """AutoRegressive Integrated Moving Average (ARIMA) model is good for predicting short-term changes in areas like customer usage or network load. 
                        \nARIMA is ideal for forecasting situations where past data, like customer usage patterns, directly influences future outcomes, such as predicting next week's network demand.
                        \n**Runtime**: The model may take 2 to 10 minutes to run.""",
            "ARIMA+ BigQuery": """An advanced version of ARIMA, perfect for working with large amounts of data. 
                        \nARIMA+ automatically fine-tunes its settings, making it great for forecasting things like nationwide data usage or large-scale network traffic.
                        \n**Runtime**: The model may take 2 to 10 minutes to run.""",
            "TimesFM": """ TimesFM from Google Cloud Platform is ideal when you need highly accurate forecasts from large datasets. It is the fastest model since it is pre-trained, allowing for rapid predictions.
                        \nTimesFM is designed for handling extensive data, making it perfect for predicting long-term trends like customer growth, network expansion needs, or service demand across multiple regions.
                        \n**Runtime**: 2 minutes"""
            
        }

        st.subheader("Choose a model")
        col1, col2 = st.columns(2)

        for idx, (model, description) in enumerate(models.items()):
            with (col1 if idx % 2 == 0 else col2):
                with st.expander(model):
                    st.write("**Description:**", description)
                    if st.button(f"Select {model}", key=model):
                        st.session_state.model_choice = model

        if st.session_state.get('model_choice') == "ARIMA":
            
            run_arima_model()
        elif st.session_state.get('model_choice') == "ARIMA+ BigQuery":
           
            run_arima_plus_model()
        
        elif st.session_state.get('model_choice') == "TimesFM":
           
            run_times_fm()
        elif st.session_state.get('model_choice') == "AutoML":
           
            run_auto_ml()
        elif st.session_state.get('model_choice') == "Prophet":
            run_prophet_training()

    elif selected == "History":
        st.title("Model History")

        # Fetch models and endpoints from Flask
        with st.spinner('Loading models and endpoints...'):
            models_response = requests.get(LIST_MODELS_URL, params={'sid': sid} , verify=os.environ.get("CERTIFICATE_PATH", False))
            endpoints_response = requests.get(LIST_ENDPOINTS_URL, params={'sid': sid} , verify=os.environ.get("CERTIFICATE_PATH", False))
            
            # models = requests.get(LIST_MODELS_URL).json().get('models', [])
            # endpoints = requests.get(LIST_ENDPOINTS_URL).json().get('endpoints', [])

        # Create two columns for Models and Endpoints
        col1, col2 = st.columns(2)
        def strip_user_id(name):
            parts = re.split('[-_]', name)
            if parts[-1].isdigit():  # If the last part is a digit, it's the user ID
                return '-'.join(parts[:-1])  # Return the name without the ID
            return name  # Return the name unchanged if no numeric suffix is found
        with col1:
            if models_response.status_code == 200:
                models = models_response.json().get('models', [])
            
                st.subheader("Your Model History")

                if not models:
                    st.info("No models found.")
                else:
                    # Display models with selection options
                    for model in models:
                        # Remove the _id suffix if present in display_name
                        display_name = strip_user_id(model["display_name"])
                        with st.expander(display_name):
                            st.write("**Status:** Not Deployed")
                            if st.button(f"Deploy {display_name}", key=f"deploy_{model['display_name']}"):
                                st.session_state.model_choice = model['display_name']
                                st.session_state.action = "deploy"
            elif models_response.status_code == 401:
                error_message = models_response.json().get('error')
                st.error(error_message)  # Show error message in Streamlit
                
                nav_script = f"""
                    <meta http-equiv="refresh" content="0; url='{os.environ.get("FLASK_SERVER_ADDR", "http://localhost:5000")}'">
                """
                st.write(nav_script, unsafe_allow_html=True)

        with col2:
            if endpoints_response.status_code == 200:
                endpoints = endpoints_response.json().get('endpoints', [])
                st.subheader("Your Endpoints")
                # Display endpoints with selection options
                if not endpoints:
                    st.info("No endpoints found.")
                else:
                    for endpoint in endpoints:
                        # Remove the _id suffix if present in displayName
                        
                        display_name = strip_user_id(endpoint["display_name"])
                        with st.expander(display_name):
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Delete {display_name}", key=f"delete_{endpoint['display_name']}"):
                                    st.session_state.action = "delete"
                                    st.session_state.endpoint_choice = endpoint['display_name']
                            with col2:
                                if st.button(f"Predict with {display_name}", key=f"predict_{endpoint['display_name']}"):
                                    st.session_state.action = "predict"
                                    st.session_state.endpoint_choice = endpoint['display_name']
            elif endpoints_response.status_code == 401:
                error_message = endpoints_response.json().get('error')
                st.error(error_message)  # Show error message in Streamlit
                
                nav_script = f"""
                    <meta http-equiv="refresh" content="0; url='{os.environ.get("FLASK_SERVER_ADDR", "http://localhost:5000")}'">
                """
                st.write(nav_script, unsafe_allow_html=True)
        # Handle the session state actions
        if st.session_state.get('action') == "deploy":
            selected_model = next(m for m in models if m['display_name'] == st.session_state.model_choice)
            # Text input for custom endpoint name
            endpoint_name = st.text_input(
                "Enter custom name for the endpoint",
                help="Provide a unique name for your endpoint. Allowed characters: letters, numbers, and underscores."
            )
            display_name = '_'.join(selected_model["display_name"].split('_')[:-1]) if '_' in model["display_name"] else selected_model["display_name"]
            if st.button(f"Deploy {display_name}", key=model):
                if selected_model and endpoint_name:
                    if not re.match(r'^[A-Za-z0-9_]+$', endpoint_name):
                        st.error("Endpoint name can only contain letters, numbers, and underscores, and must not have spaces or brackets.")
                    else:
                        # Get the resource ID using the display name
                        # model_resource_name = st.session_state.model_mapping.get(selected_model)

                        if not selected_model:
                            st.error("Could not find the model resource name. Please try again.")
                        else:
                            selected_model_id = selected_model.get('resource_name').split('/')[-1] if selected_model else None
                            deploy_model(endpoint_name, selected_model_id)
                            
             
        elif st.session_state.get('action') == "delete":
            selected_endpoint = next(e for e in endpoints if e['display_name'] == st.session_state.endpoint_choice)
            delete_endpoint(selected_endpoint)


        elif st.session_state.get('action') == "predict":
            
            selected_endpoint = next(e for e in endpoints if e['display_name'] == st.session_state.endpoint_choice)
            selected_endpoint_id = selected_endpoint.get('resource_name').split('/')[-1] if selected_endpoint else None
            endpoint_predict(selected_endpoint_id)


if __name__ == '__main__':
    main()
