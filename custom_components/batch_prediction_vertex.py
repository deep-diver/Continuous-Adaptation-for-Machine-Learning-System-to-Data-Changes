"""
This component launches a Batch Prediction job on Vertex AI.
Know more about Vertex AI Batch Predictions jobs, go here:
https://cloud.google.com/vertex-ai/docs/predictions/batch-predictions.
"""

from google.cloud import storage

from tfx.dsl.component.experimental.annotations import Parameter, InputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import String
import google.cloud.aiplatform as vertex_ai

from absl import logging


@component
def BatchPredictionGen(
    gcs_source: InputArtifact[String],
    project: Parameter[str],
    location: Parameter[str],
    model_resource_name: Parameter[str],
    job_display_name: Parameter[str],
    gcs_destination: Parameter[str],
    instances_format: Parameter[str] = "file-list",
    machine_type: Parameter[str] = "n1-standard-2",
    accelerator_count: Parameter[int] = 0,
    accelerator_type: Parameter[str] = None,
    starting_replica_count: Parameter[int] = 1,
    max_replica_count: Parameter[int] = 1,
):
    """
    gcs_source: A location inside GCS to be used by the Batch Prediction job to get its inputs.
    Rest of the parameters are explained here: https://git.io/JiUyU.
    """
    storage_client = storage.Client()

    # Read GCS Source (gcs_source contains the full path of GCS object).
    # 1-1. get bucketname from gcs_source
    gcs_source_uri = gcs_source.uri.split("//")[1:][0].split("/")
    bucketname = gcs_source_uri[0]
    bucket = storage_client.get_bucket(bucketname)
    logging.info(f"bucketname: {bucketname}")

    # 1-2. get object path without the bucket name.
    objectpath = "/".join(gcs_source_uri[1:])

    # 1-3. read the object to get value set by OutputArtifact from FileListGen.
    blob = bucket.blob(objectpath)
    logging.info(f"objectpath: {objectpath}")

    gcs_source = f"gs://{blob.download_as_text()}"

    # Get Model.
    vertex_ai.init(project=project, location=location)
    model = vertex_ai.Model.list(
        filter=f"display_name={model_resource_name}", order_by="update_time"
    )[-1]

    # Launch a Batch Prediction job.
    logging.info("Starting batch prediction job.")
    logging.info(f"GCS path where file list is: {gcs_source}")
    batch_prediction_job = model.batch_predict(
        job_display_name=job_display_name,
        instances_format=instances_format,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        accelerator_type=accelerator_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=True,
    )

    logging.info(batch_prediction_job.display_name)
    logging.info(batch_prediction_job.resource_name)
    logging.info(batch_prediction_job.state)
