"""
Generate a txt file formatted required by Vertex AI's Batch Prediction
There are few options, and this component generate "file list" formatted txt.
(https://cloud.google.com/vertex-ai/docs/predictions/batch-predictions)
"""

import tensorflow as tf
from absl import logging

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter, OutputArtifact
from tfx.types.standard_artifacts import String


@component
def FileListGen(
    outpath: OutputArtifact[String],
    gcs_src_bucket: Parameter[str],
    gcs_src_prefix: Parameter[str] = "",
    output_filename: Parameter[str] = "test-images.txt",
):
    """
    : param outpath: OutputArtifact to hold where output_filename will be located
             This will be used in the downstream component, BatchPredictionGen
    : param gcs_src_bucket: GCS bucket name where the list of raw data is
    : param gcs_src_prefix: prefix to be added to gcs_src_bucket
    : param output_filename: output filename whose content is a list of file paths of raw data
    """
    logging.info("FileListGen started")

    # 1. get the list of data
    gcs_src_prefix = (
        f"{gcs_src_prefix}/" if len(gcs_src_prefix) != 0 else gcs_src_prefix
    )
    img_paths = tf.io.gfile.glob(f"gs://{gcs_src_bucket}/{gcs_src_prefix}*.jpg")
    logging.info("Successfully retrieve the file(jpg) list from GCS path")

    # 2. write the list of data in the expected format in Vertex AI Batch Prediction to a local file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.writelines("%s\n" % img_path for img_path in img_paths)
    logging.info(
        f"Successfully created the file list file({output_filename}) in local storage"
    )

    # 3. upload the local file to GCS location
    gcs_dst = f"{gcs_src_bucket}/{gcs_src_prefix}{output_filename}"
    tf.io.gfile.copy(output_filename, f"gs://{gcs_dst}", overwrite=True)
    logging.info(f"Successfully uploaded the file list ({gcs_dst})")

    # 4. store the GCS location where the local file is
    outpath.value = gcs_dst
