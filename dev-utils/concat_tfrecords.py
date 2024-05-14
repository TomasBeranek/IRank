import tensorflow as tf

def concatenate_tfrecords(output_file, input_files):
    # Writer to write to the new TFRecord file
    with tf.io.TFRecordWriter(output_file) as writer:
        for input_file in input_files:
            # Dataset to read existing TFRecord files
            dataset = tf.data.TFRecordDataset(input_file)
            for record in dataset:
                writer.write(record.numpy())  # Write each record to the output file

# Example usage:
input_tfrecords = [ 'libtiff_0.tfrecords.test',
                    'libtiff_1.tfrecords.test']
output_tfrecord = 'libtiff-chatgpt.tfrecords.test'
concatenate_tfrecords(output_tfrecord, input_tfrecords)
