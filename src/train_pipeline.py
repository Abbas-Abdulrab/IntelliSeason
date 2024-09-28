import time
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.automl.forecasting import utils

def run_training_pipeline(project_id, location, service_account, training_dataset_bq_path, bucket_uri, target_column, date_column, identifier_column, forecast_horizon, feature_names, display_name=None):    
    # Generate display_name dynamically if not provided
    if display_name is None:
        current_time = int(time.time())  # Get the current timestamp
        display_name = f"prophet-train-{current_time}"
    
    # Log the generated display name for debugging
    print(f"Generated pipeline display name: {display_name}")

    pipeline_root = f"{bucket_uri}/{display_name}"
    root_dir = f"{bucket_uri}/pipeline_root"
    train_job_spec_path, train_parameter_values = utils.get_prophet_train_pipeline_and_parameters(
        project=project_id,
        location=location,
        root_dir=root_dir,
        time_column=date_column,
        time_series_identifier_column=identifier_column,
        target_column=target_column,
        forecast_horizon=forecast_horizon,
        optimization_objective="rmse",
        data_granularity_unit="day",
        data_source_bigquery_table_path=training_dataset_bq_path,
        window_stride_length=1,
        max_num_trials=2,
        trainer_dataflow_machine_type="n1-standard-2",
        trainer_dataflow_max_num_workers=10,
        evaluation_dataflow_machine_type="n1-standard-1",
        evaluation_dataflow_max_num_workers=1,
        dataflow_service_account=service_account,
        dataflow_use_public_ips=False
    )
    
    job = aiplatform.PipelineJob(
        job_id=display_name,
        display_name=display_name,
        pipeline_root=pipeline_root,
        template_path=train_job_spec_path,
        parameter_values=train_parameter_values,
        enable_caching=False,
    )
    # Run training job
    job.run(service_account=service_account)

    # If job has already been executed 
    # job = aiplatform.PipelineJob.get(f'projects/{project_id}/locations/{location}/pipelineJobs/{display_name}')

    # Retrieve model name
    for task_detail in job.gca_resource.job_detail.task_details:
        if task_detail.task_name == "model-upload":
            model_id = task_detail.outputs["model"].artifacts[0].metadata["resourceName"]
            break
    else:
        raise ValueError("Couldn't find the model training task.")
    
    # Add necessery labels
    add_model_labels(
        model_id=model_id,
        project_id=project_id,
        location=location,
        feature_names=feature_names,  
        target_column=target_column,
        identifier_column=identifier_column,
        date_column=date_column,
        display_name=display_name
    )

    return job


def add_model_labels(model_id, project_id, location, feature_names, target_column, identifier_column, date_column, display_name):
    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)
    
    # Get the deployed model resource
    model = aiplatform.Model(model_id)

    # Convert feature names and target column into a string format
    features = ','.join(feature_names)

    # Update model labels
    model.update(labels={
        'features': features,
        'date_column': date_column,
        'target_column': target_column,
        'identifier_column': identifier_column
    },
    display_name=display_name)

    print(f"Updated model {model.display_name} with input and output schema.")

