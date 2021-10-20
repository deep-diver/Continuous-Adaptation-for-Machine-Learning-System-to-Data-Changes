import tfx
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.types.standard_artifacts import String
from google.cloud import storage
from absl import logging


@component
def FileListGen(
    outpath: OutputArtifact[String],
    project: Parameter[str],
    gcs_source_bucket: Parameter[str],
    gcs_source_prefix: Parameter[str] = "",
    output_filename: Parameter[str] = "test-images.txt",
):
    logging.info("FileListGen started")

    client = storage.Client(project=project)
    bucket = client.get_bucket(gcs_source_bucket)
    blobs = bucket.list_blobs(prefix=gcs_source_prefix)
    logging.info("Successfully retrieve the file(jpg) list from GCS path")

    f = open(output_filename, "w")
    for blob in blobs:
        if blob.name.split(".")[-1] == "jpg":
            prefix = ""
            if gcs_source_prefix != "":
                prefix = f"/{gcs_source_prefix}"
            line = f"gs://{gcs_source_bucket}{prefix}/{blob.name}\n"
            f.write(line)
    f.close()
    logging.info(
        f"Successfully created the file list file({output_filename}) in local storage"
    )

    prefix = ""
    if gcs_source_prefix != "":
        prefix = f"{gcs_source_prefix}/"
    blob = bucket.blob(f"{prefix}{output_filename}")
    blob.upload_from_filename(output_filename)
    logging.info(f"Successfully uploaded the file list ({prefix}{output_filename})")

    outpath.value = gcs_source_bucket + "/" + prefix + output_filename
