# ******************************************************************************
#  File:            concat_tfrecords.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Script for concatenation of TFRecords files.
# ******************************************************************************

# Usage: python3 concat_tfrecords.py concatenated.tfrecords part1.tfrecords part2.tfrecords ...

import tensorflow as tf
import sys


def concatenate_tfrecords(output_file, input_files):
    # Writer to write to the new TFRecord file
    with tf.io.TFRecordWriter(output_file) as writer:
        for input_file in input_files:
            # Dataset to read existing TFRecord files
            dataset = tf.data.TFRecordDataset(input_file)
            for record in dataset:
                writer.write(record.numpy())  # Write each record to the output file

output_tfrecord = sys.argv[1]
input_tfrecords = sys.argv[2:]

concatenate_tfrecords(output_tfrecord, input_tfrecords)
