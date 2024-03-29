import tensorflow as tf
import tensorflow_gnn as tfgnn
import sys
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import sklearn

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
train_positive_samples = 371
train_negative_samples = 7325
train_ds_len = train_positive_samples + train_negative_samples
train_ds_1 = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.train'])
train_ds_0 = tf.data.TFRecordDataset([f'{project_path}_0.tfrecords.train'])
val_ds = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.val', f'{project_path}_0.tfrecords.val'])

# Up-sample
up_sample_coeff = train_negative_samples // train_positive_samples
train_ds_1 = train_ds_1.repeat(up_sample_coeff)

# Concat positive and negative samples
train_ds = train_ds_1.concatenate(train_ds_0)

# Shuffle training samples
buffer_size = train_negative_samples + up_sample_coeff * train_positive_samples # Ideally buffer_size >= len(dataset)
train_ds = train_ds.shuffle(buffer_size)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(buffer_size)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Batch the datasets
train_ds_batched = train_ds.batch(batch_size=batch_size).repeat()
val_ds_batched = val_ds.batch(batch_size=batch_size)

# Parse tf.Example protos
train_ds_batched = train_ds_batched.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
val_ds_batched = val_ds_batched.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
preproc_input_spec_train = train_ds_batched.element_spec
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

model_input_spec_train, train_ds_batched = preprocess(preproc_input_spec_train, train_ds_batched)
model_input_spec_val, val_ds_batched = preprocess(preproc_input_spec_val, val_ds_batched)

# Preprocess for validation data

# for graph, labels in train_ds_batched:
#   print(labels.numpy())
# exit()

# model_input_spec = train_ds_batched.take(1).get_single_element()[0].spec


# Encode input nodeset features into a single tensor (+ add some trainable parameters to the transformation)
def set_initial_node_state(node_set, *, node_set_name):
  # state_dims_by_node_set = {'METHOD_INFO': 32, 'TYPE': 16, 'AST_NODE': 16, 'LITERAL_VALUE': 32}
  # state_dim = state_dims_by_node_set[node_set_name]

  features = node_set.features # Immutable view

  # Combine all features to a single vector (since all of them are float32 scalars)
  stacked_features = tf.stack([v for _, v in sorted(features.items())], axis=1)
  return tf.keras.layers.Dense(node_state_dim, 'relu')(stacked_features)

def drop_all_features(_, **unused_kwargs):
  return {}

def build_model(model_input_spec):
  message_dim = node_state_dim
  state_dropout_rate = 0.05
  # l2_regularization = 1e-5

  # Input layer (takes input graphs with defined TFGNN schema)
  graph = inputs = tf.keras.layers.Input(type_spec=model_input_spec)

  # Set initial states (Encode edge states??)
  graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=set_initial_node_state,
      edge_sets_fn=drop_all_features)(graph)

  # Layers of updates
  for i in range(num_graph_updates):
    graph = mt_albis.MtAlbisGraphUpdate(
        units=node_state_dim,
        message_dim=message_dim,
        receiver_tag=tfgnn.SOURCE, # Up the AST tree
        # edge_feature_name='ARGUMENT_INDEX',
        node_set_names=None if i < num_graph_updates-1 else ["AST_NODE"],
        simple_conv_reduce_type="mean|sum",
        state_dropout_rate=state_dropout_rate,
        # l2_regularization=l2_regularization,
        normalization_type="layer",
        next_state_type="residual" # Input layer must have same size of HIDDEN_STATE as units
    )(graph)

    # graph = mt_albis.MtAlbisGraphUpdate(
    #     units=node_state_dim,
    #     message_dim=message_dim,
    #     receiver_tag=tfgnn.TARGET, # Up the AST tree
    #     # edge_feature_name='ARGUMENT_INDEX',
    #     simple_conv_reduce_type="mean|sum",
    #     state_dropout_rate=state_dropout_rate,
    #     l2_regularization=l2_regularization,
    #     normalization_type="layer",
    #     next_state_type="residual" # Input layer must have same size of HIDDEN_STATE as units
    # )(graph)

  # Read hidden states from AST_NODE nodeset
  readout_features = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "max", node_set_name=['AST_NODE'])(graph)

  # Add 'head' - final part GNN which outputsrw-rw-r--  1 tomas tomas  128033122 bře 19 01:20 libtiff_0.tfrecords.testngle number
  y = tf.keras.layers.Dense(1, activation='sigmoid')(readout_features)
  return tf.keras.Model(inputs, y)

# Define Loss and Metrics
# with tf.device('/CPU:0'):  # Forces the operations to be executed on the CPU
model = build_model(model_input_spec_train)

loss = tf.keras.losses.BinaryCrossentropy()
metrics = [ tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr')]
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the keras model
model.compile(optimizer, loss=loss, metrics=metrics)
model.summary()

# Calculate weights to balance the data
# weight_for_positive = train_ds_len / (2.0 * train_positive_samples)
# weight_for_negative = train_ds_len / (2.0 * train_negative_samples)
# class_weight = {0: weight_for_negative, 1: weight_for_positive}

early_stopping = EarlyStopping(monitor='val_auc',
                               min_delta=0.001,
                               patience=5,
                               restore_best_weights=True)

# Train the model
history = model.fit(train_ds_batched,
                    steps_per_epoch=train_ds_len // batch_size,
                    epochs=300,
                    validation_data=val_ds_batched,
                    # class_weight=class_weight,
                    shuffle=True,
                    callbacks=[early_stopping])

model.save('saved_models/1')

def extract_graph(graph, label):
    return graph

def extract_label(graph, label):
    return label

# Remove label from dataset for prediction
val_ds_batched_graphs = val_ds_batched.map(extract_graph)
val_ds_batched_graph = val_ds_batched.map(extract_label)

label_pred = model.predict(val_ds_batched_graphs)

# Convert tensors with batches to a simple list
label_gt = []
for batch in val_ds_batched_graph:
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
  fpr, tpr, thresholds = sklearn.metrics.roc_curve(label_gt, label_pred)

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
