# Time Series Forecasting API

This project is a Python 3-based API for time series forecasting using Google Cloud Platforms, Flask, & Streamlit. It supports multiple forecasting models, including Prophet, Arima, Arima+, TimesFM, AutoML. Users can upload a `dataset.csv` file, specify the column to predict, and select the forecasting model to use. The API will process the data and return the predictions.

## Features

- **Upload Dataset**: Users can upload a CSV file containing the time series data.
- **Model Selection**: Users can choose from various forecasting models, including Prophet.
- **Column Specification**: Users specify which column in the dataset they want to predict.
- **Predictions**: The API processes the data and returns the forecasted values.

## Requirements

- Python 3.11
- Flask for creating the API
- Streamlit
- Requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forecasting-api.git
   cd forecasting-api

## Install the required packages:
`pip install -r requirements.txt`

## Start the API:
`streamlit run landing.py --server.port 8502`

## Upload a Dataset:
Interact with the streamlit UI to upload csv to model.

### Example curl command:

`curl -X POST "http://localhost:5000/predict" \
-F "file=@path/to/your/dataset.csv" \
-F "column=your_column_name" \
-F "model=prophet"`

## Get Predictions:
The API will return the forecasted values based on the specified model.

## API Endpoints
- For AutoML, an endpoint must be deployed within the streamlit app.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or suggestions, please contact GenZ.
