import tensorflow as tf
import tensorflow_gnn as tfgnn
import sys
from tensorflow_gnn import runner
from tensorflow_gnn.models import mt_albis
from tensorflow.keras.metrics import AUC


# Hyperparameters
learning_rate = 0.0001
batch_size = 12
node_state_dim = 16
num_graph_updates = 8


# Load args
schema_path = sys.argv[1]
project_path = sys.argv[2]

# Load schema
graph_schema = tfgnn.read_schema(schema_path)
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


# def decode_fn(record_bytes):
#     graph = tfgnn.parse_single_example(graph_tensor_spec, record_bytes, validate=True)

#     # Extract label (GT) from context and remove it from graph
#     context_features = graph.context.get_features_dict()
#     label = int(context_features.pop('label'))
#     new_graph = graph.replace_features(context=context_features)

#     return new_graph, label

# Read the dataset
train_positive_samples = 371
train_negative_samples = 7325
train_ds_len = train_positive_samples + train_negative_samples
train_ds = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.train', f'{project_path}_0.tfrecords.train'])
val_ds = tf.data.TFRecordDataset([f'{project_path}_1.tfrecords.val', f'{project_path}_0.tfrecords.val'])

# Shuffle training samples
buffer_size = 1000
train_ds = train_ds.shuffle(buffer_size)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# train_ds_serialized = train_ds.map(lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))
# for i, graph in enumerate(train_ds_serialized.take(1000)):
#   print(f"Input {i}: {graph.context.features['label'].numpy()}")
# exit()

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
  return model_input_spec, ds_batched, labels

model_input_spec_train, train_ds_batched, train_labels = preprocess(preproc_input_spec_train, train_ds_batched)
model_input_spec_val, val_ds_batched, val_labels = preprocess(preproc_input_spec_val, val_ds_batched)

# Preprocess for validation data

# for graph, labels in train_ds_batched.take(5):
#   print(labels.numpy())

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
  # state_dropout_rate = 0.2
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
        # node_set_names=None if i < num_graph_updates-1 else ["AST_NODE"],
        simple_conv_reduce_type="mean|sum",
        # state_dropout_rate=state_dropout_rate,
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
  readout_features = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "mean", node_set_name=['METHOD_INFO', 'TYPE', 'AST_NODE', 'LITERAL_VALUE'])(graph)

  # Add 'head' - final part GNN which outputs a single logit
  logit = tf.keras.layers.Dense(1)(readout_features)
  return tf.keras.Model(inputs, logit)


# Define Loss and Metrics
# with tf.device('/CPU:0'):  # Forces the operations to be executed on the CPU
model = build_model(model_input_spec_train)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.),
            tf.keras.metrics.BinaryCrossentropy(from_logits=True),
            AUC()]
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the keras model
model.compile(optimizer, loss=loss, metrics=metrics)
model.summary()

# Calculate weights to balance the data
weight_for_positive = train_ds_len / (2.0 * train_positive_samples)
weight_for_negative = train_ds_len / (2.0 * train_negative_samples)
class_weight = {0: weight_for_negative, 1: weight_for_positive}

# Train the model
history = model.fit(train_ds_batched,
                    steps_per_epoch=train_ds_len // batch_size,
                    epochs=20,
                    validation_data=val_ds_batched,
                    class_weight=class_weight,
                    shuffle=True)

# Plot the loss and metric curves for train and val
# for k, hist in history.history.items():
#   plt.plot(hist)
#   plt.title(k)
#   plt.show()
