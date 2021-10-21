"""
Component responsible for triggering a training job given a pipeline specification.
"""

import json

from google.cloud import storage

from kfp.v2.google.client import AIPlatformClient
from tfx.dsl.component.experimental.annotations import Parameter, InputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.types.experimental.simple_artifacts import Dataset

from absl import logging


@component
def PipelineTrigger(
    is_retrain: InputArtifact[Dataset],
    latest_span_id: InputArtifact[Dataset],
    pipeline_spec_path: Parameter[str],
    project_id: Parameter[str],
    region: Parameter[str],
):
    """
    :param is_retrain: Boolean to indicate if we are retraining.
    :param latest_span_id: Latest span id to craft training data for the model.
    :param pipeline_spec_path: Training pipeline specification path.
    :param project_id: GCP project id.
    :param region: GCP region.
    """
    if is_retrain.get_string_custom_property("result") == "False":
        # Check if the pipeline spec exists.
        storage_client = storage.Client()

        path_parts = pipeline_spec_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])

        bucket = storage_client.bucket(bucket_name)
        blob = storage.Blob(bucket=bucket, name=blob_name)

        if not blob.exists(storage_client):
            raise ValueError(f"{pipeline_spec_path} does not exist.")

        # Initialize Vertex AI API client and submit for pipeline execution.
        api_client = AIPlatformClient(project_id=project_id, region=region)

        # Fetch the latest span.
        latest_span = latest_span_id.get_string_custom_property("latest_span")

        # Create a training job from pipeline spec.
        response = api_client.create_run_from_job_spec(
            pipeline_spec_path,
            enable_caching=False,
            parameter_values={
                "input-config": json.dumps(
                    {
                        "splits": [
                            {
                                "name": "train",
                                "pattern": f"span-[{int(latest_span)-1}{latest_span}]/train/*.tfrecord",
                            },
                            {
                                "name": "val",
                                "pattern": f"span-[{int(latest_span)-1}{latest_span}]/test/*.tfrecord",
                            },
                        ]
                    }
                ),
                "output-config": json.dumps({}),
            },
        )
        logging.info(response)
