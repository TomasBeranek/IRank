import tensorflow as tf
import sys


def count_samples_in_tfrecord(tfrecord_file):
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_file):
        count += 1
    return count

if __name__ == '__main__':
    tfrecord_file = sys.argv[1]
    number_of_samples = count_samples_in_tfrecord(tfrecord_file)
    print(f'Number of samples: {number_of_samples}')
