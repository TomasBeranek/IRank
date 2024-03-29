import tensorflow as tf
import tensorflow_gnn as tfgnn
import sys
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Hyperparameters
learning_rate = 0.000001
batch_size = 12
node_state_dim = 16
num_graph_updates = 8

# Load args
schema_path = sys.argv[1]
project_path = sys.argv[2]

# Load schema
graph_schema = tfgnn.read_schema(schema_path)
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

# Read the dataset
val_ds = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.val', f'{project_path}_0.tfrecords.val'])

# val_ds_1 = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.val']).take(3)
# val_ds_0 = tf.data.TFRecordDataset([f'{project_path}_0.tfrecords.val']).take(118)
# val_ds = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.val', f'{project_path}_0.tfrecords.val'])

# Concat positive and negative samples
# val_ds = val_ds_1.concatenate(val_ds_0)

val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Batch the datasets
val_ds_batched = val_ds.batch(batch_size=batch_size)

# Parse tf.Example protos
val_ds_batched = val_ds_batched.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
preproc_input_spec_val = val_ds_batched.element_spec

# Define preprocess model which will ONLY extract labels out of graph
def preprocess(preproc_input_spec, ds_batched):
  preproc_input = tf.keras.layers.Input(type_spec=preproc_input_spec)
  graph = preproc_input.merge_batch_to_components()
  labels = tfgnn.keras.layers.Readout(from_context=True, feature_name='label')(graph)
  graph = graph.remove_features(context=["label"])
  assert "label" not in graph.context.features
  preproc_model = tf.keras.Model(preproc_input, (graph, labels))
  ds_batched = ds_batched.map(preproc_model)
  model_input_spec, _ = ds_batched.element_spec # We dont need spec for labels
  return model_input_spec, ds_batched

model_input_spec_val, val_ds_batched = preprocess(preproc_input_spec_val, val_ds_batched)

# Load model
model = tf.keras.models.load_model('saved_models/1')

def extract_graph(graph, label):
    return graph

def extract_label(graph, label):
    return label

# Remove label from dataset for prediction
val_ds_batched_graphs = val_ds_batched.map(extract_graph)
val_ds_batched_labels = val_ds_batched.map(extract_label)

label_pred = model.predict(val_ds_batched_graphs)

# Convert tensors with batches to a simple list
label_gt = []
for batch in val_ds_batched_labels:
   label_gt.extend(batch.numpy().tolist())


# Convert model predictions to a simple list
label_pred = label_pred.flatten().tolist()

# Sort it according to the highest probability to be TP (likeliest TP is first)
labels_sorted = sorted(zip(label_pred, label_gt), key=lambda x: x[0], reverse=True)

def plot_top_N_precision(labels_sorted):
  y = []
  x = list(range(2, 101)) # Skip 1% as it gets extreme values for libtiff
  labels_sorted = [label[1] for label in labels_sorted]
  base_precision = sum(labels_sorted) / len(labels_sorted)
  captions = []

  for percentage in x:
    top_items_cnt = round(len(labels_sorted) * (percentage / 100))
    TP_percentage = sum(labels_sorted[:top_items_cnt]) / top_items_cnt
    y.append(TP_percentage)
    captions.append(top_items_cnt)

  # Plot the graph
  fig, ax = plt.subplots()
  ax.plot(x, y)
  for i, txt in enumerate(captions):
    ax.text(x[i], y[i], txt, fontsize=9)
  ax.set_title("My Plot")
  ax.set_xlabel("Top N% of most likely TPs")
  ax.set_ylabel("TP percentage (Precision)")
  plt.axhline(y=base_precision, color='r', linestyle=':')
  plt.axvline(x=base_precision * 100, color='r', linestyle=':')
  plt.show()


def plot_AUC_curve(label_gt, label_pred):
  fpr, tpr, thresholds = metrics.roc_curve(label_gt, label_pred)

  # Calculate the AUC
  roc_auc = metrics.auc(fpr, tpr)

  # Plot the ROC curve
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()


plot_top_N_precision(labels_sorted)
plot_AUC_curve(label_gt, label_pred)
