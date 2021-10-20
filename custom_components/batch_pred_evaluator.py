# Reference: https://bit.ly/vertex-batch

from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.decorators import component
from tfx.types.experimental.simple_artifacts import Dataset

from absl import logging
import os
import json


@component
def PerformanceEvaluator(
    gcs_destination: Parameter[str],
    local_directory: Parameter[str],
    threshold: Parameter[float],
    trigger_pipeline: OutputArtifact[Dataset],
):
    full_gcs_results_dir = f"{gcs_destination}/{local_directory}"

    # Create missing directories.
    os.makedirs(local_directory, exist_ok=True)

    # Get the Cloud Storage paths for each result.
    os.system(f"gsutil -m cp -r {full_gcs_results_dir} {local_directory}")

    # Get most recently modified directory.
    latest_directory = max(
        [os.path.join(local_directory, d) for d in os.listdir(local_directory)],
        key=os.path.getmtime,
    )

    # Get downloaded results in directory.
    results_files = []
    for dirpath, subdirs, files in os.walk(latest_directory):
        for file in files:
            if file.startswith("prediction.results"):
                results_files.append(os.path.join(dirpath, file))

    # Consolidate all the results into a list.
    results = []
    for results_file in results_files:
        # Download each result.
        with open(results_file, "r") as file:
            results.extend([json.loads(line) for line in file.readlines()])

    # Calculate performance.
    num_correct = 0

    for result in results:
        label = os.path.basename(result["instance"]).split("_")[0]
        prediction = result["prediction"]["label"]

        if label == prediction:
            num_correct = num_correct + 1

    accuracy = num_correct / len(results)
    logging.info(f"Accuracy: {accuracy*100}%")
    trigger_pipeline.set_string_custom_property("result", str(accuracy >= threshold))
