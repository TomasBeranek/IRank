import tensorflow as tf
import tensorflow_gnn as tfgnn
import sys
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os

# Train script for mixed nodes models
from mixed_nodes_model import preprocess


def find_latest_model_dir(models_dir):
  entries = os.listdir(models_dir)
  entries_sorted = sorted(entries)
  return os.path.join(models_dir, entries_sorted[-1])


def construct_model_dir_from_id(models_dir, model_id):
  for entry in os.listdir(models_dir):
    if entry.startswith(f'{model_id}_AUC_'):
      return os.path.join(models_dir, entry)


def extract_graph(graph, label):
    return graph


def extract_label(graph, label):
    return label


def load_dataset(datasets_path, project):
  # Read the dataset
  val_ds = tf.data.TFRecordDataset([os.path.join(datasets_path, f'{project}_1.tfrecords.val'),
                                    os.path.join(datasets_path, f'{project}_0.tfrecords.val')])
  val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

  # Batch the datasets
  batch_size = 12
  val_ds_batched = val_ds.batch(batch_size=batch_size)

  # Parse tf.Example protos
  val_ds_batched = val_ds_batched.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
  preproc_input_spec_val = val_ds_batched.element_spec

  # Apply preprocess model
  _, val_ds_batched = preprocess(preproc_input_spec_val, val_ds_batched)

  # Remove label from dataset for prediction
  val_ds_batched_graphs = val_ds_batched.map(extract_graph)
  val_ds_batched_labels = val_ds_batched.map(extract_label)

  return val_ds_batched_graphs, val_ds_batched_labels


def plot_top_N_precision(labels_sorted_dict, models_dir):
  fig, ax = plt.subplots()
  colors = {'httpd': 'red', 'libtiff': 'green', 'nginx': 'blue'}
  for project, labels_sorted in labels_sorted_dict.items():
    color = colors[project]
    y = []
    x = list(range(2, 101)) # Calculate percentages from 2% to 100%
    labels_sorted = [label[1] for label in labels_sorted]
    base_precision = sum(labels_sorted) / len(labels_sorted)
    captions = []

    for percentage in x:
      top_items_cnt = round(len(labels_sorted) * (percentage / 100))
      TP_percentage = sum(labels_sorted[:top_items_cnt]) / top_items_cnt
      y.append(TP_percentage)
      captions.append(top_items_cnt)

    # Plot the graph
    ax.plot(x, y, label=project, color=color)
    for i, txt in enumerate(captions):
      ax.text(x[i], y[i], txt, fontsize=8, color=color)
    plt.axhline(y=base_precision, color=color, linestyle=':')
    plt.axvline(x=base_precision * 100, color=color, linestyle=':')

  ax.set_title("Top N% precision")
  ax.set_xlabel("Top N% of most likely TPs")
  ax.set_ylabel("TP percentage (Precision)")
  plt.legend()
  plt.savefig(os.path.join(models_dir, 'Top_N_precisions.png'))
  plt.show()


def plot_AUC_curve(label_pair_dicts, models_dir):
  plt.figure()
  colors = {'httpd': 'red', 'libtiff': 'green', 'nginx': 'blue'}

  for project, (label_pred, label_gt) in label_pair_dicts.items():
    color = colors[project]

    fpr, tpr, _ = metrics.roc_curve(label_gt, label_pred)

    # Calculate the AUC
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, label='%s AUC %0.2f' % (project, roc_auc))

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(models_dir, 'AUC_curves.png'))
  plt.show()


if __name__ == '__main__':
  # Load args
  schema_path = sys.argv[1]
  datasets_path = sys.argv[2]
  model_id = sys.argv[3]
  saved_models_dir = 'saved_models'

  if model_id == '--latest':
    models_dir = find_latest_model_dir(saved_models_dir)
  else:
    models_dir = construct_model_dir_from_id(saved_models_dir, model_id)

  # Load schema
  graph_schema = tfgnn.read_schema(schema_path)
  graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  labels_sorted_dict = {}
  label_pairs_dict = {}

  # Evalute each of the httpd, libtiff and nginx models
  for model_dir in os.listdir(models_dir):
    if model_dir.startswith('httpd_AUC_'):
      project = 'httpd'
    elif model_dir.startswith('libtiff_AUC_'):
      project = 'libtiff'
    elif model_dir.startswith('nginx_AUC_'):
      project = 'nginx'
    else:
      continue

    # Prepare data
    val_ds_batched_graphs, val_ds_batched_labels = load_dataset(datasets_path, project)

    # Load model
    model = tf.keras.models.load_model(os.path.join(models_dir, model_dir))

    label_pred = model.predict(val_ds_batched_graphs)

    # Convert tensors with batches to a simple list
    label_gt = []
    for batch in val_ds_batched_labels:
      label_gt.extend(batch.numpy().tolist())

    # Convert model predictions to a simple list
    label_pred = label_pred.flatten().tolist()

    label_pairs_dict[project] = label_pred, label_gt

    # Sort it according to the highest probability to be TP (likeliest TP is first)
    labels_sorted_dict[project] = sorted(zip(label_pred, label_gt), key=lambda x: x[0], reverse=True)

  plot_top_N_precision(labels_sorted_dict, models_dir)
  plot_AUC_curve(label_pairs_dict, models_dir)