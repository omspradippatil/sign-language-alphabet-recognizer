import sys
import os
from pathlib import Path


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

image_path = sys.argv[1]


def resolve_artifact(primary_path, fallback_path, artifact_name):
    primary = Path(primary_path)
    fallback = Path(fallback_path)
    if primary.exists():
        return str(primary)
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(
        "Missing {}. Expected '{}' or '{}'.".format(artifact_name, primary_path, fallback_path)
    )


labels_path = resolve_artifact("logs/output_labels.txt", "logs/trained_labels.txt", "labels file")
graph_path = resolve_artifact("logs/output_graph.pb", "logs/trained_graph.pb", "graph file")

# Read the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels_path)]

# Unpersists graph from file
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
