import tensorflow as tf
import numpy as np
import pickle, os
from collections import deque

classes = ['Opaque','Red','Green','Yellow','Bright','Light Blue','Colors','Red',
            'Women','Enemy','Son','Man','Away','Drawer','Born','Learn',
            'Call','Skimmer','Bitter','Sweet Milk']

def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label[l.strip()] = count
        count += 1
    return label

def get_data(input_data_dump, num_frames_per_video, labels):
  X, y = [], []
  temp_list = deque()
  frames = []

  if input_data_dump is None:
    with open(os.getcwd()+'/predicted-frames-final_result-test.pkl','rb') as fin:
      frames = pickle.load(fin)
  else:
    frames = np.array(input_data_dump)

  print('Shape',np.array(frames).shape)
  for i,frame in enumerate(frames):
    features = frame
  #   actual = frame[1].lower()

  #   actual = labels[actual]

    if len(temp_list) == num_frames_per_video-1:
      temp_list.append(features)
      flat = list(temp_list)
      X.append(np.array(flat))
      # y.append(actual)
      temp_list.clear()
    else:
      temp_list.append(features)
      continue

#   print("Class Name\tNumeric Label")
#   for key in labels:
#     print("%s\t\t%d" % (key, labels[key]))

  X = np.asarray(frames)
#   y = np.asarray(y)

  print("Dataset shape: ", X.shape)

#   y = np.eye(len(labels))[y]

  print('dis?')
  return X,[]#y


def predictor(input_data=None, num_frames_per_video=101, batch_size=100, labels=load_labels('retrained_labels.txt'), model_file='model_sign.h5'):
    X,Y = get_data(input_data, num_frames_per_video, labels)
    print(X.shape)
    num_classes = len(labels)
    size_of_each_frame = X.shape[2]

    model = tf.keras.models.load_model(model_file)
    predictions = model.predict(X)
    print(predictions)
    predictions = np.array([np.argmax(pred) for pred in predictions])
    Y = np.array([np.argmax(each) for each in Y])

    print(Y)
    print('\n\n\t\t',str(predictions[0]+1),classes[predictions[0]],'\n\n')

# main(model_file='model_sign.h5')