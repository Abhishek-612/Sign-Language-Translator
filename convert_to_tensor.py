import os,sys
import tqdm
import tensorflow as tf
import numpy as np
import pickle

model_file = 'retrained_graph.pb'
input_layer = 'Placeholder'
output_layer = 'final_result'

NUM_PARALLEL_EXEC_UNITS = 1
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
                        inter_op_parallelism_threads=4, 
                        allow_soft_placement=True,
                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})

def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(frames, input_height=299, input_width=299, input_mean=0, input_std=255):
    # input_name = "file_reader"
    # frames = [(tf.read_file(frame, input_name), frame) for frame in frames]
    # print(tf.image.convert_image_dtype(frames, dtype=tf.uint8, saturate=False, name=None))
    decoded_frames = [tf.convert_to_tensor(frame, dtype=tf.uint8) for frame in frames]
    float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]
    float_caster = tf.stack(float_caster)
    resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session(config=config)
    result = sess.run(normalized)
    return result

def predict(graph, image_tensor):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph,config=config) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )
    results = np.squeeze(results)
    return results

def predict_on_frames(frames, batch_size=100,save_file=True):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    batch_size = batch_size
    graph = load_graph()
    
    # frames = [each for each in os.walk(frames_folder) if os.path.basename(each[0]) in labels_in_dir]

    predictions = []
    batch = frames
    
    try:
        frames_tensors = read_tensor_from_image_file(batch, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
        pred = predict(graph, frames_tensors)
        # pred = [[each.tolist(), os.path.basename(label)] for each in pred]
        predictions.extend(pred)

    except KeyboardInterrupt:
        print("You quit with ctrl+c")
        sys.exit()

    except Exception as e:
        print("Error making prediction: %s" % (e))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if save_file:
        out_file = os.getcwd()+'/predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], 'test')
        print("Dumping predictions to: %s" % (out_file))
        with open(out_file, 'wb') as fout:
            pickle.dump(np.array([predictions]), fout)
            print('um')

        print("Done.")
    # return predictions

#DELETE
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("graph", help="graph/model to be executed")
#     parser.add_argument("frames_folder", help="'Path to folder containing folders of frames of different gestures.'")
#     parser.add_argument("--input_layer", help="name of input layer", default='Placeholder')
#     parser.add_argument("--output_layer", help="name of output layer", default='final_result')
#     parser.add_argument('--test', action='store_true', help='passed if frames_folder belongs to test_data')
#     parser.add_argument("--batch_size", help="batch Size", default=10)
#     args = parser.parse_args()

#     model_file = args.graph if args.graph else 'retrained_graph.pb'
#     frames_folder = args.frames_folder
#     input_layer = args.input_layer
#     output_layer = args.output_layer
#     batch_size = int(args.batch_size)

#     if args.test:
#         train_or_test = "test"
#     else:
#         train_or_test = "train" # Was train

#     # reduce tf verbosity
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)

#     out_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], train_or_test)
#     print("Dumping predictions to: %s" % (out_file))
#     with open(out_file, 'wb') as fout:
#         pickle.dump(predictions, fout)

#     print("Done.")