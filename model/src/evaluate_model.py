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

colors = {'httpd': 'red',
          'libtiff': 'green',
          'nginx': 'blue',
          'combined': 'grey',
          'libav': 'purple',
          'Model 8': 'blue',
          'Model 10': 'red',
          'Model 13': 'green',
          '3-soft-vote': 'purple',
          '6-soft-vote': 'orange',
          'chatgpt': 'teal'}

def construct_model_dir_from_id(models_dir, model_id):
  for entry in os.listdir(models_dir):
    if entry.startswith(f'{model_id}_AUC_'):
      return os.path.join(models_dir, entry)


def extract_graph(graph, label):
    return graph


def extract_label(graph, label):
    return label


def load_dataset(graph_tensor_spec, datasets_path, project, data_type):
  # Read the dataset
  if project == 'combined':
    val_ds = tf.data.TFRecordDataset([os.path.join(datasets_path, f'httpd_1.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'httpd_0.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'libtiff_1.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'libtiff_0.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'nginx_1.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'nginx_0.tfrecords.{data_type}')])
  elif project == 'libav':
    val_ds = tf.data.TFRecordDataset([os.path.join(datasets_path, f'libav_1.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'libav_0.tfrecords.{data_type}')])
  elif project == 'chatgpt':
    val_ds = tf.data.TFRecordDataset([os.path.join(datasets_path, f'libtiff-chatgpt.tfrecords.test')])
  else:
    val_ds = tf.data.TFRecordDataset([os.path.join(datasets_path, f'{project}_1.tfrecords.{data_type}'),
                                      os.path.join(datasets_path, f'{project}_0.tfrecords.{data_type}')])
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


def plot_top_N_precision(labels_sorted_dict, data_type):
  _, ax = plt.subplots()
  print('Top N% Precision')
  for project, labels_sorted in labels_sorted_dict.items():
    print(f'\t{project}')
    color = colors[project]
    y = []
    x = list(range(2, 101)) # Calculate percentages from 2% to 100%
    labels_sorted = [label[1] for label in labels_sorted]
    base_precision = sum(labels_sorted) / len(labels_sorted)
    captions = []

    for percentage in x:
      top_items_cnt = round(len(labels_sorted) * (percentage / 100))
      TP_count = sum(labels_sorted[:top_items_cnt])
      TP_percentage = TP_count / top_items_cnt
      y.append(TP_percentage)
      print(f'\t\t{percentage} % ({top_items_cnt}): Precision: {round(TP_percentage, 2)} Number of TPs: {TP_count}')

    # Plot the graph
    ax.plot(x, y, label=project, color=color)
    plt.axhline(y=base_precision, color=color, linestyle=':')

  ax.set_title(f"Top N% Precision ({data_type})")
  ax.set_xlabel("Top N% Samples")
  ax.set_ylabel("Precision")
  plt.legend()
  plt.savefig('Top_N_precisions.svg', format='svg')
  plt.show()


def plot_ROC_curve(label_pair_dicts, data_type):
  plt.figure()

  for project, (label_pred, label_gt) in label_pair_dicts.items():
    color = colors[project]

    # Calculate the AUC
    fpr, tpr, _ = metrics.roc_curve(label_gt, label_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, label='%s AUC %0.2f' % (project, roc_auc))

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'Receiver Operating Characteristic ({data_type})')
  plt.legend(loc="lower right")
  plt.savefig('ROC_curves.svg', format='svg')
  plt.show()


def scenario(schema_path, datasets_path, saved_models_dir, data_type, dataset):
  labels_sorted_dict = {}
  label_pairs_dict = {}

  voting3_gt = {}
  voting3_pred = {}

  voting6_gt = {}
  voting6_pred = {}

  for model_id in [8, 10, 13, 6, 11, 12]:
    project = f'Model {model_id}'
    model_dir = construct_model_dir_from_id(saved_models_dir, model_id)

    # Load schema
    graph_schema = tfgnn.read_schema(schema_path)
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    # Get correct model path
    for sub_model_dir in os.listdir(model_dir):
      if sub_model_dir.startswith('combined_AUC_'):
        model_dir = os.path.join(model_dir, sub_model_dir)
        break

    val_ds_batched_graphs, val_ds_batched_labels = load_dataset(graph_tensor_spec, datasets_path, dataset, data_type)

    # Load model and make predictions
    model = tf.keras.models.load_model(model_dir)
    label_pred = model.predict(val_ds_batched_graphs)

    # Convert tensors with batches to a simple list
    label_gt = []
    for batch in val_ds_batched_labels:
      label_gt.extend(batch.numpy().tolist())

    # Convert model predictions to a simple list
    label_pred = label_pred.flatten().tolist()

    if model_id in [8, 10, 13]:
      # Make a copy of results for 3-voting model
      voting3_gt[model_id] = list(label_gt)
      voting3_pred[model_id] = np.array(label_pred)

    # Make a copy of results for 6-voting model
    voting6_gt[model_id] = list(label_gt)
    voting6_pred[model_id] = np.array(label_pred)

    # Dont plot results for these models, they are used only for soft voting
    if model_id in [6, 11, 12]:
      continue

    # Sort it according to the highest probability to be TP (likeliest TP is first)
    labels_sorted_dict[project] = sorted(zip(label_pred, label_gt), key=lambda x: x[0], reverse=True)
    label_pairs_dict[project] = label_pred, label_gt

  # Create soft 3-voting score
  soft_vote3_pred = list(voting3_pred.values())
  soft_vote3_pred = np.sum(soft_vote3_pred, axis=0).tolist()
  labels_sorted_dict['3-soft-vote'] = sorted(zip(soft_vote3_pred, voting3_gt[8]), key=lambda x: x[0], reverse=True)
  label_pairs_dict['3-soft-vote'] = soft_vote3_pred, voting3_gt[8]

  # Create soft 6-voting score
  soft_vote6_pred = list(voting6_pred.values())
  soft_vote6_pred = np.sum(soft_vote6_pred, axis=0).tolist()
  labels_sorted_dict['6-soft-vote'] = sorted(zip(soft_vote6_pred, voting6_gt[8]), key=lambda x: x[0], reverse=True)
  label_pairs_dict['6-soft-vote'] = soft_vote6_pred, voting6_gt[8]

  if dataset == 'chatgpt':
      # Drop all results except 6-soft-vote
      label_pairs_dict = {k: v for k, v in label_pairs_dict.items() if k == '6-soft-vote'}

      # ChatGPT results
      label_pairs_dict['chatgpt'] = ([0.95, 0.95, 1, 1, 0.85, 0.8, 0.9, 0.8, 0.82, 0.95, 1, 0.95, 0.9, 0.95, 0.95, 1, 0.95, 1, 0.95, 0.85],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

      plot_ROC_curve(label_pairs_dict, f'{data_type}-{dataset}')
      return

  plot_top_N_precision(labels_sorted_dict, f'{data_type}-{dataset}')
  plot_ROC_curve(label_pairs_dict, f'{data_type}-{dataset}')


if __name__ == '__main__':
  # Load args
  schema_path = sys.argv[1]
  datasets_path = sys.argv[2]
  saved_models_dir = sys.argv[3]
  model_id = sys.argv[4]
  data_type = sys.argv[5] # test or val
  saved_models_dir = 'saved_models'

  if model_id in ['combined', 'httpd', 'libtiff', 'nginx', 'libav', 'chatgpt']:
    # We run predefined scenario
    scenario(schema_path, datasets_path, saved_models_dir, data_type, model_id)
    exit()
  else:
    # Evaluate single models
    model_id = int(model_id)

  # Model 6 and higher are trained on combined data
  if model_id < 6:
    combined = False
  else:
    combined = True

  model_dir = construct_model_dir_from_id(saved_models_dir, model_id)

  # Load schema
  graph_schema = tfgnn.read_schema(schema_path)
  graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  if combined:
    # Get correct model path
    for sub_model_dir in os.listdir(model_dir):
      if sub_model_dir.startswith('combined_AUC_'):
        model_dir = os.path.join(model_dir, sub_model_dir)
        break

    val_ds_batched_graphs, val_ds_batched_labels = load_dataset(graph_tensor_spec, datasets_path, 'combined', data_type)

    # Load model and make predictions
    model = tf.keras.models.load_model(model_dir)
    label_pred = model.predict(val_ds_batched_graphs)

    # Convert tensors with batches to a simple list
    label_gt = []
    for batch in val_ds_batched_labels:
      label_gt.extend(batch.numpy().tolist())

    # Convert model predictions to a simple list
    label_pred = label_pred.flatten().tolist()

    # Sort it according to the highest probability to be TP (likeliest TP is first)
    labels_sorted = sorted(zip(label_pred, label_gt), key=lambda x: x[0], reverse=True)

    plot_top_N_precision({'combined': labels_sorted}, data_type)
    plot_ROC_curve({'combined': (label_pred, label_gt)}, data_type)
  else:
    labels_sorted_dict = {}
    label_pairs_dict = {}

    for sub_model_dir in os.listdir(model_dir):
      if sub_model_dir.startswith('httpd_AUC_'):
        project = 'httpd'
      elif sub_model_dir.startswith('libtiff_AUC_'):
        project = 'libtiff'
      elif sub_model_dir.startswith('nginx_AUC_'):
        project = 'nginx'
      else:
        continue

      print(f'Project: {project}')

      # Prepare data
      val_ds_batched_graphs, val_ds_batched_labels = load_dataset(graph_tensor_spec, datasets_path, project, data_type)

      # Load model a make predictions
      model = tf.keras.models.load_model(os.path.join(model_dir, sub_model_dir))
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

    plot_top_N_precision(labels_sorted_dict, data_type)
    plot_ROC_curve(label_pairs_dict, data_type)
