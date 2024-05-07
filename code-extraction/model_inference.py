import tensorflow as tf
import tensorflow_gnn as tfgnn
import sys
import json

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


def extract_graph(graph, label):
    return graph


def extract_label(graph, label):
    return label


def load_dataset(graph_tensor_spec, dataset_path):
  # Read the dataset
  ds = tf.data.TFRecordDataset([dataset_path])
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

  # Batch the datasets
  batch_size = 8
  ds_batched = ds.batch(batch_size=batch_size)

  # Parse tf.Example protos
  ds_batched = ds_batched.map(tfgnn.keras.layers.ParseExample(graph_tensor_spec))
  preproc_input_spec = ds_batched.element_spec

  # Apply preprocess model
  _, ds_batched = preprocess(preproc_input_spec, ds_batched)

  # Remove label from dataset for prediction
  ds_batched_graphs = ds_batched.map(extract_graph)
  ds_batched_labels = ds_batched.map(extract_label)

  return ds_batched_graphs, ds_batched_labels


if __name__ == '__main__':
  # Load args
  schema_path = sys.argv[1]
  dataset_path = sys.argv[2]
  model_dir = sys.argv[3]
  infer_json = sys.argv[4]
  ranked_infer_json = sys.argv[5]

  # Load schema
  graph_schema = tfgnn.read_schema(schema_path)
  graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

  # Load model
  model = tf.keras.models.load_model(model_dir)

  # Preprocess data
  ds_batched_graphs, ds_batched_labels = load_dataset(graph_tensor_spec, dataset_path)

  # Make predictions
  scores = model.predict(ds_batched_graphs)

  # Convert tensors with batches to a simple list
  sample_ids = []
  for batch in ds_batched_labels:
    sample_ids.extend(batch.numpy().tolist())

  # Convert model predictions to a simple list
  scores = scores.flatten().tolist()

  # Load Infer's report.json
  with open(infer_json, 'r') as file:
    original_reports = json.load(file)

  reports_without_scores = []
  reports_with_scores = []
  for id, report in enumerate(original_reports):
    if id not in sample_ids:
      # Score is missing for this sample
      reports_without_scores.append(report)
      continue

    idx = sample_ids.index(id)
    report['model_score'] = scores[idx]
    reports_with_scores.append(report)

  # Sort reports according to score and append reports without scores at the end
  ranked_reports = sorted(reports_with_scores, key=lambda x: x['model_score'], reverse=True)
  all_ranked_reports = ranked_reports + reports_without_scores

  # Save the ranked reports in JSON
  with open(ranked_infer_json, 'w') as file:
    json.dump(all_ranked_reports, file, indent=4)
