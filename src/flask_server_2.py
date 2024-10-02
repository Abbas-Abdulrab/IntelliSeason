import os
from flask import Flask, request, redirect, session, jsonify, render_template, url_for
from requests_oauthlib import OAuth2Session
from google.oauth2.credentials import Credentials
from google.cloud import bigquery
from google.auth.transport.requests import Request
from google.cloud import storage, aiplatform
from google.cloud.exceptions import NotFound
import subprocess
import json
import os
import pandas as pd
import requests
from io import StringIO
from google.protobuf import json_format
import time
from datetime import datetime
from google.auth.exceptions import RefreshError
import logging
import re
import io
from train_pipeline import run_training_pipeline


UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')
# Store the token in memory
global_token = None
CREDENTIALS = None
user_info = None
global_model_response = None
streamlit_process = None
app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')

client_id = os.environ.get('CLIENT_ID')
client_secret = os.environ.get('CLIENT_SECRET')
authorization_base_url = os.environ.get('AUTHORIZATION_BASE_URL')
token_url = os.environ.get('TOKEN_URL')
redirect_uri = os.environ.get('REDIRECT_URI')
bucket_name = os.environ.get('BUCKET_NAME')
user_info_url = os.environ.get('USER_INFO_URL')
PROJECT_ID = os.environ.get('PROJECT_ID')
DATA_SET_ID = os.environ.get('DATA_SET_ID')
PROJECT_EP = os.environ.get('PROJECT_EP')
SERVICE_ACCOUNT = os.environ.get('SERVICE_ACCOUNT')
TRAIN_JOB = os.environ.get('TRAIN_JOB')
PREDICT_URL_TEMPLATE = os.environ.get('PREDICT_URL_TEMPLATE')
LOCATION = os.environ.get('LOCATION')
scope = [
    'https://www.googleapis.com/auth/userinfo.profile', 
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/devstorage.full_control',
    'https://www.googleapis.com/auth/bigquery'
]

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    
def get_or_refresh_token():
    global global_token
    if not global_token:
        return {"status_code": 401, "error": "Session expired. Please log in again."}

    # Use the existing global_token to create credentials
    credentials = Credentials(
        token=global_token['access_token'],
        refresh_token=global_token['refresh_token'],
        token_uri=token_url,  # Token URL to refresh the token
        client_id=client_id,
        client_secret=client_secret,
    )

    # Check if the credentials are expired and refresh if necessary
    if credentials.expired:
        try:
            credentials = credentials.refresh(Request())
            # Update the global token and session with the new access token
            global_token = credentials.token
        
            # session['oauth_token'] = global_token  # Update the session
        except Exception as e:
            return {"status_code": 401, "error": "Failed to refresh token. Please log in again."}

    return {"status_code": 200, "credentials": credentials}

@app.route('/')
def index():
    google = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
    
    authorization_url, state = google.authorization_url(authorization_base_url, access_type="offline")
    session['oauth_state'] = state
    return redirect(authorization_url)


@app.route('/streamlit')
def streamlit_app():
    return redirect("http://localhost:8501")

@app.route('/callback')
def callback():
    global global_token
    global user_info
    global streamlit_process

    google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=session['oauth_state'])
    token = google.fetch_token(token_url, client_secret=client_secret, authorization_response=request.url)
    
    # Set the global token and store it in the session to persist across requests
    global_token = token
    session['oauth_token'] = token

    user_info = google.get(user_info_url).json()

    # Save user info in session
    session['user_info'] = {
        'id': user_info.get('id'),
        'name': user_info.get('name'),
        'email': user_info.get('email')
    }

    # Only start Streamlit if it hasn't been started yet
    if not streamlit_process or streamlit_process.poll() is not None:
        env = os.environ.copy()
        env['BROWSER'] = 'none'
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "streamlit2.py", "--server.headless", "true"],
            env=env
        )

    return redirect("/streamlit")

def get_user_info():
    response = get_or_refresh_token()
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
        
    access_token = credentials.token
    headers = {'Authorization': f'Bearer {access_token}'}
    userinfo_response = requests.get(user_info_url, headers=headers)
    return user_info
# Function to handle date parsing
def parse_date(date_series):
    possible_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", 
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%SZ",
    ]

    def try_parse_single_date(date_str):
        for fmt in possible_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"No valid date format found for: {date_str}")

    # Apply to each element in the series
    return date_series.apply(try_parse_single_date)

@app.route('/automl',methods=['POST'])
def automl():
    global user_info
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    file_path = request.form.get('file_path')
    target_column = request.form.get('target_column')
    date_column = request.form.get('date_column')

    if not file_path or not os.path.exists(file_path):
        return "Invalid file path", 400
    
    storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    # Upload the CSV file to the bucket
    blob_name = f"{os.path.basename(file_path).split('.')[0]}_{user_info['id']}.csv"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    # Initialize Vertex AI client
    aiplatform.init(project=PROJECT_ID, location='europe-west1', credentials=credentials)

    # Create dataset
    dataset = aiplatform.TabularDataset.create(
        display_name=f"{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
        gcs_source=[f"gs://{bucket_name}/{blob_name}"]
    )

    job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"training_job_{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
            optimization_prediction_type="regression",
            optimization_objective="minimize-rmse"
        )

    # Train the model
    model_display_name=f"model_{os.path.basename(file_path).split('.')[0]}_{user_info['id']}"
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=1000,
        model_display_name=f"model_{os.path.basename(file_path).split('.')[0]}_{user_info['id']}",
        disable_early_stopping=False
    )

    return jsonify({"model_display_name": model_display_name})
    # return f"Successfully uploaded {os.path.basename(file_path)} to Google Cloud Storage and started AutoML training."
    
@app.route('/upload_data', methods=['POST'])
def upload_data():
    global user_info
    if not user_info:
        user_info = get_user_info()

    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500

    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    try:
        
      
        # Retrieve JSON data from request
        request_data = request.json
        train_file_path = request_data.get('train_file_path')
       
        if not train_file_path:
            return jsonify({"error": "Both train and test data are required."}), 400

        # Initialize BigQuery client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        train_base_file_name = os.path.splitext(os.path.basename(train_file_path))[0]
        train_table_id = f"{PROJECT_ID}.{DATA_SET_ID}.{train_base_file_name}_{user_info['id']}"

        # Define job configuration with auto-detection
        job_config = bigquery.LoadJobConfig(
            autodetect=True,  # Enable schema auto-detection
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1
        )

        # Load Train CSV to BigQuery
        with open(train_file_path, "rb") as train_file:
            train_job = client.load_table_from_file(train_file, train_table_id, job_config=job_config)
            train_job.result()  # Wait for the job to complete

        
        # Rearrange the table by timestamp

        query = f"""
            CREATE OR REPLACE TABLE `{train_table_id}` AS
            SELECT * FROM `{train_table_id}`
            ORDER BY DATE(Date) ASC;
        """
        query_job = client.query(query)

        # query_job.result()  # Wait for the job to complete

        return jsonify({"message": "Data uploaded successfully."}), 200

    except Exception as e:
        # Log the error message
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Endpoint to handle ARIMA+ forecasting
@app.route('/run_arima_plus', methods=['GET'])
def run_arima_plus():
    global user_info
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
   
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    try:
         # Retrieve query parameters
        date_column = request.args.get('date_column')
        target_column = request.args.get('target_column')
        forecast_period = int(request.args.get('forecast_period', 5))
        train_file_path = request.args.get('train_file_path')
       
        if not date_column or not target_column:
            return jsonify({"error": "Date column and target column are required."}), 400
        base_file_name = os.path.splitext(os.path.basename(train_file_path))[0]
        train_table_id = f"{PROJECT_ID}.{DATA_SET_ID}.{base_file_name}_{user_info['id']}"
       
        # Create or Replace ARIMA model
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_plus_model`
        OPTIONS(
            model_type='ARIMA_PLUS',
            time_series_data_col='{target_column}',
            time_series_timestamp_col='{date_column}',
            horizon={forecast_period},
            auto_arima_max_order=5,
            TIME_SERIES_LENGTH_FRACTION = 0.2,
            SEASONALITIES = ['DAILY', 'WEEKLY', 'MONTHLY', 'YEARLY']
        ) AS
        SELECT
            {date_column},
            
            {target_column}
          
        FROM
            `{train_table_id}`;
        """
        client.query(create_model_query).result()

        # Forecast using ARIMA_PLUS model
        forecast_query = f"""
        SELECT
            *
        FROM
            ML.EXPLAIN_FORECAST(MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_plus_model`, STRUCT({forecast_period} AS horizon));
        """
        forecast_df = client.query(forecast_query).to_dataframe()

        # Delete the ARIMA Plus model after forecasting
        delete_model_query = f"DROP MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_plus_model`"
        client.query(delete_model_query).result()


        # Return forecast results as JSON
        return jsonify(forecast_df.to_dict(orient='records'))
    
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/run_arima', methods=['GET'])
def run_arima():
    # Replace the global token with your actual token management approach
    global user_info
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    try:
         # Retrieve query parameters
        date_column = request.args.get('date_column')
        target_column = request.args.get('target_column')
        forecast_period = int(request.args.get('forecast_period', 5))
        train_file_path = request.args.get('train_file_path')
        if not date_column or not target_column:
            return jsonify({"error": "Date column and target column are required."}), 400
        base_file_name = os.path.splitext(os.path.basename(train_file_path))[0]
        train_table_id = f"{PROJECT_ID}.{DATA_SET_ID}.{base_file_name}_{user_info['id']}"

        # Create or Replace ARIMA model
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_model`
        OPTIONS(
            model_type='ARIMA',
            time_series_data_col='{target_column}',
            time_series_timestamp_col='{date_column}',
            horizon={forecast_period}
        ) AS
        SELECT
            {date_column},
            {target_column}
        FROM
            `{train_table_id}`;
        """
        client.query(create_model_query).result()

        # Forecast using ARIMA_PLUS model
        forecast_query = f"""
        SELECT
            *
        FROM
            ML.FORECAST(MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_model`, STRUCT({forecast_period} AS horizon));
        """
        forecast_df = client.query(forecast_query).to_dataframe()

        # Delete the ARIMA model after forecasting
        delete_model_query = f"DROP MODEL `{PROJECT_ID}.{DATA_SET_ID}.arima_model`"
        client.query(delete_model_query).result()

        # Return forecast results as JSON
        return jsonify(forecast_df.to_dict(orient='records'))
    
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500




@app.route('/model', methods=['POST'])
def model():
    # Replace the global token with your actual token management approach
    global user_info    
    
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
        
    global_bearer_token = credentials.token
    # Retrieve CSV data, date column, and target column from the request
    csv_data = request.json.get('csv_data')
    date_column = request.json.get('date_column')
    target_column = request.json.get('target_column')

    if not csv_data or not date_column or not target_column:
        return jsonify({"error": "CSV data, date column, or target column not specified."}), 400

    try:
        # Load the CSV data into a DataFrame from the in-memory string using io.StringIO
        df = pd.read_csv(StringIO(csv_data))
        df = df[[date_column, target_column]].dropna()

        # Process the date column
        df[date_column] = parse_date(df[date_column])

        if df[date_column].isnull().any():
            unparsed_dates = df[df[date_column].isnull()]

        df.dropna(subset=[date_column], inplace=True)
        df.set_index(date_column, inplace=True)

        # Prepare the data for the model
        data = df[target_column].tolist()

    except Exception as e:
        return jsonify({"error": f"Error processing CSV file: {e}"}), 500

    # Define a function to create payloads
    def create_payloads(data, timestamps=None, horizon=None, freq=None, chunk_size=500):
        payloads = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]

            # Construct the instance dictionary
            instance = {
                "input": chunk  # The required time-series data
            }

            # Add optional fields only if they are provided
            if timestamps:
                instance["timestamp"] = timestamps[i:i + chunk_size]
            if horizon is not None:
                instance["horizon"] = horizon
            if freq is not None:
                instance["freq"] = freq

            # Add the instance to the payload
            payload = json.dumps({"instances": [instance]})
            payloads.append(payload)
        return payloads
    payloads = create_payloads(data)

    model_url = PREDICT_URL_TEMPLATE
    headers = {
        'Authorization': f'Bearer {global_bearer_token}',
        'Content-Type': 'application/json'
    }

    global global_model_response
    global_model_response = []
    for payload in payloads:
        response = requests.post(model_url, headers=headers, data=payload)
        if response.status_code == 200:
            global_model_response.append(response.json())
        else:
            return jsonify({"error": f'Failed to retrieve model response. Status code: {response.status_code}, Response: {response.text}'}), 500

    return jsonify({"message": "Model run completed successfully."})

@app.route('/get_model_response')
def get_model_response():
    # Replace the global token with your actual token management approach
    global user_info
    
    credentials = get_or_refresh_token()
    global global_model_response
    if not global_model_response:
        return jsonify({"error": "No data available"}), 404
    return jsonify(global_model_response)


# Function to get tables based on specific ID pattern
@app.route('/get_tables',methods=['GET'])
def get_tables_with_id():
    
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500

    # credentials.refresh(Request())
    client = bigquery.Client(credentials=credentials, project="genz-forecast-project")
   
    query = f"""
    SELECT
        table_name
    FROM
        `{PROJECT_ID}.{DATA_SET_ID}.INFORMATION_SCHEMA.TABLES`; 
    """
    query_job = client.query(query, location=LOCATION)
    table_list = []
    
    for table in query_job:
        table_list.append(table)
    # return table_list
    return [row.table_name for row in query_job]

#########################
'''
prophet - maather

'''
@app.route('/predict_prophet', methods=['POST'])
def predict_prophet():
    # Replace the global token with your actual token management approach
    global user_info

    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    global_bearer_token = credentials.token
    # Check for authentication
    if not global_bearer_token:
        return jsonify({"error": "Not authenticated"}), 401

    # Extract data from the incoming request
    request_data = request.get_json()


    # Ensure the request data is valid
    if not request_data:
        return jsonify({"error": "No data provided in request."}), 400

    # Extract 'endpoint_id' and 'instances' from the incoming request
    endpoint_id = request_data.get('endpoint_id')
    instances = request_data.get('instances')

    # Check if both instances and endpoint_id are provided
    if not instances or not endpoint_id:
        return jsonify({"error": "No instances or endpoint ID provided"}), 400

    if not endpoint_id.startswith('projects/'):
        endpoint_id = f'projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}'

    endpoint_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1/{endpoint_id}:predict'

    headers = {
        "Authorization": f"Bearer {global_bearer_token}",
        "Content-Type": "application/json"
    }

    payload = {"instances": instances}

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload)
        if response.status_code == 200:
            return jsonify(response.json())  # Return the model's prediction response
        else:
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
'''
predict/automl/endpoints - Abbas
'''

@app.route('/list_models')
def list_models():

    response = get_or_refresh_token()
   
    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    
    global user_info
    if not user_info:
        user_info = get_user_info()

    try:
        # Ensure a valid token is available
        if global_token is None:
            return jsonify({"error": "Authentication required. Please click the button below to authenticate."}), 401

        user_id = user_info.get('id')  # This field contains the unique user ID
        if not user_id:
            return jsonify({"error": "User ID not found in user info."}), 500
        
        # Initialize AI Platform with the credentials
        aiplatform.init(credentials=credentials, project=PROJECT_ID, location=LOCATION)

        # List models and filter by user ID
        models = aiplatform.Model.list()
        
        filtered_models = [
            {"display_name": model.display_name, "resource_name": model.resource_name}
            for model in models if user_id in model.display_name  # Adjust this filter based on your use case
        ]

        return jsonify({"models": filtered_models}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to list models: {e}"}), 500

@app.route('/deploy_model', methods=['POST'])
def deploy_model():
    try:
        model_name = request.json.get('model_name')
        custom_endpoint_name = request.json.get('endpoint_name')

        if not model_name or not custom_endpoint_name:
            return jsonify({"error": "Model name and custom endpoint name are required"}), 400

        if not re.match(r'^[A-Za-z0-9_]+$', custom_endpoint_name):
            return jsonify({"error": "Endpoint name can only contain letters, numbers, and underscores, and must not have spaces or brackets."}), 400

        # Fetch user information to get the user ID
        if global_token is None:
            return jsonify({"error": "Authentication required."}), 401

        # Fetch user information using the access token
        access_token = global_token.get('access_token')
        if not access_token:
            return jsonify({"error": "Failed to retrieve access token."}), 500

        # Call Google's UserInfo endpoint to get user information
        userinfo_endpoint = 'https://www.googleapis.com/oauth2/v1/userinfo'
        headers = {'Authorization': f'Bearer {access_token}'}
        userinfo_response = requests.get(userinfo_endpoint, headers=headers)

        if userinfo_response.status_code != 200:
            return jsonify({"error": "Failed to fetch user information from token."}), 500

        user_info = userinfo_response.json()
        user_id = user_info.get('id')  # This field contains the unique user ID
        if not user_id:
            return jsonify({"error": "User ID not found in user info."}), 500

        # Initialize AI Platform
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # List existing endpoints for the user
        existing_endpoints = aiplatform.Endpoint.list()
        user_endpoints = [ep for ep in existing_endpoints if user_id in ep.display_name]

        # Check for duplicate endpoint name (excluding user ID)
        for ep in user_endpoints:
            # Extract the custom name by removing '_userID' from the display name
            if ep.display_name.endswith(f"_{user_id}"):
                existing_custom_name = ep.display_name[:-len(f"_{user_id}")]
                if existing_custom_name == custom_endpoint_name:
                    return jsonify({"error": f"You already have an endpoint with the name '{custom_endpoint_name}'. Please choose a different name."}), 400

        # Append user ID to the endpoint name
        custom_endpoint_name += f"_{user_id}"

        # Check if the user already has 2 endpoints
        if len(user_endpoints) >= 2:
            return jsonify({"error": "You can only have 2 endpoints. Please delete an existing endpoint before deploying a new one."}), 400

        # Proceed with deploying the model
        model = aiplatform.Model(model_name=model_name)
        endpoint = aiplatform.Endpoint.create(display_name=custom_endpoint_name)
        deployed_model = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f"{custom_endpoint_name}_deployed_model",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1
        )

        deployed_model.wait()

        endpoint_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{endpoint.resource_name}"

        endpoint_info = {
            "endpoint_id": endpoint.name.split('/')[-1],
            "endpoint_display_name": endpoint.display_name,
            "endpoint_url": endpoint_url,
            "status": "Deployment successful"
        }

        return jsonify(endpoint_info), 200

    except Exception as e:
        return jsonify({"error": f"Failed to deploy model: {e}"}), 500

@app.route('/list_user_endpoints')
def list_user_endpoints():
    # global global_token
    
    global user_info
    if not user_info:
        user_info = get_user_info()
    try:
        
        response = get_or_refresh_token()

        # Ensure response is a dictionary and contains the 'status_code' key
        if isinstance(response, dict) and 'status_code' in response:
            if response['status_code'] == 200:
                credentials = response['credentials']
                # Continue with your BigQuery operations
            else:
                # Return an error response to the client or handle it as needed
                return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
        else:
            # Handle cases where the response is not what we expected
            return jsonify({"error": "Unexpected response format."}), 500
                
        global_token = credentials.token
        access_token = global_token
        if not access_token:
            return jsonify({"error": "Failed to retrieve access token."}), 500

        user_id = user_info.get('id')  # Extract the user ID from the response

        if not user_id:
            return jsonify({"error": "User ID not found in user info."}), 500

        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        existing_endpoints = aiplatform.Endpoint.list()
        user_endpoints = [
            {"display_name": ep.display_name, "resource_name": ep.resource_name}
            for ep in existing_endpoints #if user_id in ep.display_name
        ]

        return jsonify({"endpoints": user_endpoints}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to list user endpoints: {e}"}), 500

@app.route('/delete_endpoint', methods=['POST'])
def delete_endpoint():
    try:
        endpoint_name = request.json.get('endpoint_name')
        if not endpoint_name:
            return jsonify({"error": "Endpoint name is required."}), 400

        # Initialize AI Platform
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # Get the endpoint object
        endpoint = aiplatform.Endpoint(endpoint_name)
        deployed_models = endpoint.gca_resource.deployed_models

        # Undeploy all models on the endpoint
        if deployed_models:
            for model in deployed_models:
                endpoint.undeploy(model.id)

        # Wait for a brief moment to ensure undeployment completes
        time.sleep(5)  # Adjust the time as necessary for your context

        # Delete the endpoint
        endpoint.delete()

        return jsonify({"status": "Endpoint deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to delete endpoint: {e}"}), 500

def get_deployed_model_id(endpoint_resource_name):
    try:
        # Initialize the AI Platform client
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # Get the endpoint object
        endpoint = aiplatform.Endpoint(endpoint_resource_name)

        # List deployed models on the endpoint
        deployed_models = endpoint.gca_resource.deployed_models

        # Check for deployed models and return the ID
        if deployed_models:
            for model in deployed_models:
                logging.info(f"Found deployed model ID: {model.id}")
            return deployed_models[0].id  # Assuming there is at least one deployed model
        else:
            logging.info("No models are deployed on this endpoint.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving deployed model ID: {e}")
        return None



@app.route('/list_endpoints', methods=['GET'])
def list_endpoints():
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    global_bearer_token = credentials.token
    """Fetches the list of deployed endpoints with models from GCP AI Platform."""
    if not global_bearer_token:
        return jsonify({"error": "Not authenticated"}), 401
    
 
    url = f'https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints'
    
    headers = {
        "Authorization": f"Bearer {global_bearer_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        endpoints = response.json().get('endpoints', [])
        # Filter endpoints to include only those with models
        endpoints_with_models = [endpoint for endpoint in endpoints if endpoint.get('deployedModels')]
        
        return jsonify({"endpoints": endpoints_with_models})
    else:
        return jsonify({"error": response.text}), response.status_code


@app.route('/get_columns', methods=['GET'])
def get_columns():
    global user_info
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    global_bearer_token = credentials.token
    """Fetches the required input schema for the given endpoint."""
    endpoint_id = request.args.get('endpoint_id')
    if not global_bearer_token or not endpoint_id:
        return jsonify({"error": "Not authenticated or no endpoint ID provided"}), 401
    
    url = f'https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{endpoint_id}'

    headers = {
        "Authorization": f"Bearer {global_bearer_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        deployed_models = response.json().get('deployedModels', [])
        if not deployed_models:
            return jsonify({"error": "No models deployed to this endpoint."}), 404


        # Extract input columns from explanationSpec
        explanation_metadata = deployed_models[0].get('explanationSpec', {}).get('metadata', {})
        if explanation_metadata:
            inputs = explanation_metadata.get('inputs', {})
            if inputs:
                columns = list(inputs.keys())  # Extract the column names from the 'inputs' dictionary
                return jsonify({"columns": columns, "source": "metadata"})
                # return jsonify({"columns": columns})
            
        # Fallback: If no columns found in explanationSpec, proceed to check the model details for labels
        model_id = deployed_models[0].get('model')
        model_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1/{model_id}'
        
        # Fetch model details to locate the labels
        model_response = requests.get(model_url, headers=headers)
        
        if model_response.status_code == 200:
            model_details = model_response.json()
            labels = model_details.get('labels', {})
            if labels:
                # Extract columns from the labels
                target_column = labels.get('target_column')
                identifier_column = labels.get('identifier_column')
                features = labels.get('features', '')

                # Parse features if it's a comma-separated string
                feature_columns = features.split(',') if features else []
                all_columns = {'target_column': [target_column], 'identifier_column': [identifier_column], 'feature_columns': feature_columns}
                
                # Return categorized columns
                # Return categorized columns and label source
                return jsonify({"categorized_columns": all_columns, "source": "labels"})
                # return jsonify({"categorized_columns": all_columns})
            else:
                return jsonify({"error": "No labels found in model details."}), 404
        else:
            return jsonify({"error": model_response.text}), model_response.status_code
    else:
        return jsonify({"error": response.text}), response.status_code


@app.route('/predict', methods=['POST'])
def predict():
    response = get_or_refresh_token()

    # Ensure response is a dictionary and contains the 'status_code' key
    if isinstance(response, dict) and 'status_code' in response:
        if response['status_code'] == 200:
            credentials = response['credentials']
            # Continue with your BigQuery operations
        else:
            # Return an error response to the client or handle it as needed
            return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
    else:
        # Handle cases where the response is not what we expected
        return jsonify({"error": "Unexpected response format."}), 500
    
    global_bearer_token = credentials.token
    if not global_bearer_token:
        return jsonify({"error": "Not authenticated"}), 401
    
    # Parse JSON data correctly
    request_data = request.get_json()
    data = request_data.get('data')
    endpoint_id = request_data.get('endpoint_id')

    if not data or not endpoint_id:
        return jsonify({"error": "No data or endpoint ID provided"}), 400
    
    # Ensure the endpoint ID is in the correct format
    project_id = "genz-forecast-project"
    location = 'europe-west1'
    
    if not endpoint_id.startswith('projects/'):
        # Construct the full resource name for the endpoint
        endpoint_id = f'projects/{PROJECT_ID}/locations/{location}/endpoints/{endpoint_id}'
    # Construct the prediction URL
    endpoint_url = f'https://{location}-aiplatform.googleapis.com/v1/{endpoint_id}:predict'
    headers = {
        "Authorization": f"Bearer {global_bearer_token}",
        "Content-Type": "application/json"
    }
    
    # Make the prediction request to GCP AI Platform
    response = requests.post(endpoint_url, headers=headers, json={"instances": data})
    
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": response.text}), response.status_code

###############################################
@app.route('/upload_data_prophet', methods=['POST'])
def upload_data_prophet():
    global user_info
    global global_token

    if global_token is None or user_info is None:
        return jsonify({"error": "User not authenticated"}), 401

    try:
        data = request.get_json()
        csv_data = data.get('csv_data')
        date_column = data.get('date_column')
        identifier_column = data.get('identifier_column')

        if not all([csv_data, date_column, identifier_column]):
            return jsonify({"error": "Missing parameters"}), 400

        # Convert CSV data back to DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Initialize BigQuery client


        response = get_or_refresh_token()
        if isinstance(response, dict) and 'status_code' in response:
            if response['status_code'] == 200:
                credentials = response['credentials']
                # Continue with your BigQuery operations
            else:
                # Return an error response to the client or handle it as needed
                return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
        else:
            # Handle cases where the response is not what we expected
            return jsonify({"error": "Unexpected response format."}), 500
        client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

        # Name of the table in BigQuery where data will be uploaded
        dataset_name = "training_data"
        user_id = user_info['id']
        table_name = f"uploaded_{int(time.time())}_{user_id}"

        # Step 1: Upload CSV to BigQuery
        dataset_ref = bigquery.DatasetReference(PROJECT_ID, dataset_name)
        table_ref = dataset_ref.table(table_name)
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            source_format=bigquery.SourceFormat.CSV,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        # Upload the DataFrame to BigQuery
        load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        load_job.result()  # Wait for the job to complete

        # Step 2: Rearrange the table by timestamp (sorting)
        query = f"""
            CREATE OR REPLACE TABLE `{PROJECT_ID}.{dataset_name}.{table_name}` AS
            SELECT * FROM `{PROJECT_ID}.{dataset_name}.{table_name}`
            ORDER BY DATE({date_column}) ASC;
        """
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete

        return jsonify({"status": "Data uploaded and sorted", "table_name": table_name}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Prophet
@app.route('/process_and_train', methods=['POST'])
def process_and_train():
    global user_info
    global global_token

    if global_token is None or user_info is None:
        return jsonify({"error": "User not authenticated"}), 401

    try:
        data = request.get_json()
        table_name = data.get('table_name')  # Get the table name from the previous function
        target_column = data.get('target_column')
        date_column = data.get('date_column')
        identifier_column = data.get('identifier_column')
        forecast_horizon = data.get('forecast_horizon')
        feature_names = data.get('feature_names')

        if not all([table_name, target_column, date_column, identifier_column, forecast_horizon, feature_names]):
            return jsonify({"error": "Missing parameters"}), 400

        # Initialize BigQuery client
        
        response = get_or_refresh_token()
        if isinstance(response, dict) and 'status_code' in response:
            if response['status_code'] == 200:
                credentials = response['credentials']
                # Continue with your BigQuery operations
            else:
                # Return an error response to the client or handle it as needed
                return jsonify({"error": response.get('error', 'Unknown error occurred')}), 401
        else:
            # Handle cases where the response is not what we expected
            return jsonify({"error": "Unexpected response format."}), 500
        client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

        dataset_name = "training_data"

        # Step 3: Add rows for missing identifiers and fill target column with 0
        query_fill_missing = f"""
            CREATE OR REPLACE TABLE `{PROJECT_ID}.{dataset_name}.{table_name}` AS
            WITH unique_dates AS (
                SELECT DISTINCT {date_column} AS date
                FROM `{PROJECT_ID}.{dataset_name}.{table_name}`
            ),
            unique_identifiers AS (
                SELECT DISTINCT {identifier_column} AS identifier
                FROM `{PROJECT_ID}.{dataset_name}.{table_name}`
            ),
            all_combinations AS (
                SELECT date, identifier
                FROM unique_dates
                CROSS JOIN unique_identifiers
            )
            SELECT
                all_combinations.date AS {date_column},
                all_combinations.identifier AS {identifier_column},
                COALESCE(CAST(t.{target_column} AS FLOAT64), 0) AS {target_column},
                t.* EXCEPT({date_column}, {identifier_column}, {target_column})
            FROM all_combinations
            LEFT JOIN `{PROJECT_ID}.{dataset_name}.{table_name}` t
            ON all_combinations.date = t.{date_column}
            AND all_combinations.identifier = t.{identifier_column}
            ORDER BY {date_column}, {identifier_column};
        """
        query_job_fill = client.query(query_fill_missing)
        query_job_fill.result()  # Wait for the query to complete

        # Initialize the Vertex AI SDK
        aiplatform.init(
            project=PROJECT_ID,
            location=LOCATION,
            credentials=credentials
        )

        # Step 4: Run the training pipeline
        training_table_path = f"{PROJECT_ID}.{dataset_name}.{table_name}"
        display_name = f"prophet-train-{int(time.time())}-{user_info['id']}"
        BUCKET_NAME = 'genz_v1'
        BUCKET_URI = f"gs://{BUCKET_NAME}"
        # Call your training pipeline function here
        # Assuming run_training_pipeline is defined elsewhere
        job = run_training_pipeline(
            project_id=PROJECT_ID,
            location=LOCATION,
            service_account=SERVICE_ACCOUNT,
            training_dataset_bq_path=training_table_path,
            bucket_uri=BUCKET_URI,
            target_column=target_column,
            date_column=date_column,
            identifier_column=identifier_column,
            forecast_horizon=forecast_horizon,
            feature_names=feature_names,
            display_name=display_name
        )

        return jsonify({"status": "Training pipeline started", "job_id": job.name}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
if __name__ == '__main__':
    app.debug = True
    app.run()