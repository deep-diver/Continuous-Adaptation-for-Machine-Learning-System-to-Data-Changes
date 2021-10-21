"""
This component is responsible for separating provided samples into training and
validation splits. It then converts them to TFRecords and stores those inside
a GCS location. Finally, it returns the latest span id calculated from the current
samples in `gcs_source_bucket`.
"""

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.annotations import OutputArtifact, InputArtifact
from tfx.types.experimental.simple_artifacts import Dataset
from absl import logging

from datetime import datetime
import tensorflow as tf
import random
import os


# Label-mapping.
LABEL_DICT = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}


# Images are byte-strings.
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


# Classes would be integers.
def _int_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


# Function that prepares a record for the tfrecord file
# a record contains the image and its label.
def to_tfrecord(img_bytes, label):
    feature = {
        "image": _bytestring_feature([img_bytes]),
        "label": _int_feature([label]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(filepaths, dest_gcs, tfrecord_filename, new_span, is_train):
    # For this project, we are serializing the images in one TFRecord only.
    # For more realistic purposes, this should be sharded.
    folder = "train" if is_train else "test"

    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for path in filepaths:
            image_string = tf.io.read_file(path).numpy()
            class_name = path.split("/")[-1].split("_")[0]
            label = LABEL_DICT[class_name]

            example = to_tfrecord(image_string, label)
            writer.write(example.SerializeToString())

    # Copy over the zipped TFRecord file to the GCS Bucket and
    # remove the temporary files.
    logging.info(f"gsutil cp {tfrecord_filename} {dest_gcs}/span-{new_span}/{folder}/")
    os.system(f"gsutil cp {tfrecord_filename} {dest_gcs}/span-{new_span}/{folder}/")
    os.remove(tfrecord_filename)


@component
def SpanPreparator(
    is_retrain: InputArtifact[Dataset],
    gcs_source_bucket: Parameter[str],
    gcs_destination_bucket: Parameter[str],
    latest_span_id: OutputArtifact[Dataset],
    gcs_source_prefix: Parameter[str] = "",
):
    """
    :param is_retrain: Boolean to indicate if we are retraining.
    :param gcs_source_bucket: GCS location where the entry samples are residing.
    :param gcs_destination_bucket: GCS location where the converted TFRecords will be serialized.
    :param latest_span_id: Data span.
    :param gcs_source_prefix: Location prefix.
    """
    if is_retrain.get_string_custom_property("result") == "False":
        # Get the latest span and determine the new span.
        last_span_str = tf.io.gfile.glob(f"{gcs_destination_bucket}/span-*")[-1]
        last_span = int(last_span_str.split("-")[-1])
        new_span = last_span + 1

        timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")

        # Get images from the provided GCS source.
        image_paths = tf.io.gfile.glob(f"gs://{gcs_source_bucket}/*.jpg")
        logging.info(image_paths)
        random.shuffle(image_paths)

        # Create train and validation splits.
        val_split = 0.2
        split_index = int(len(image_paths) * (1 - val_split))
        training_paths = image_paths[:split_index]
        validation_paths = image_paths[split_index:]

        # Write as TFRecords.
        write_tfrecords(
            training_paths,
            gcs_destination_bucket,
            tfrecord_filename=f"new_training_data_{timestamp}.tfrecord",
            new_span=new_span,
            is_train=True,
        )
        write_tfrecords(
            validation_paths,
            gcs_destination_bucket,
            tfrecord_filename=f"new_validation_data_{timestamp}.tfrecord",
            new_span=new_span,
            is_train=False,
        )

        logging.info("Removing images from batch prediction bucket.")
        os.system(
            f"gsutil mv gs://{gcs_source_bucket}/{gcs_source_prefix} gs://{gcs_source_bucket}/{gcs_source_prefix}_old"
        )
        latest_span_id.set_string_custom_property("latest_span", str(new_span))
