#######################################################################
############################## Packages ###############################
#######################################################################

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import cv2
import glob
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as img
from IPython.display import Image, display, clear_output

#######################################################################
################ Initialize Functions/Variables #######################
#######################################################################

#!!!!!!!!!!!!!!!!!!!!!!!!!! Make changes to lines: 44, 88, 98, 118, 119

# Science Thresholds

person_threshold = 0.50
person_gun_threshold = 0.80


# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sys.path.append("..")

os.chdir("/models-master/research/object_detection/")
# ## Object detection imports
# Here are the imports from the object detection module.

# Needed if you want to make bounded boxes around person for object detection
from utils import label_map_util
from utils import visualization_utils as vis_util

##################### Model Preparation ###############################

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08' #original line


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box and select person class
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1

#Load a Object Detection Model(frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Object Recognition model
#label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")] #original line
label_lines = [line.rstrip() for line in tf.gfile.GFile("C:/pistol-detection/tf_files/retrained_labels.txt")] #edited line


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    #with tf.gfile.FastGFile('/tf_files/retrained_graph.pb', 'rb') as h: #original line
    with tf.gfile.FastGFile('C:/pistol-detection/tf_files/retrained_graph.pb', 'rb') as h: #edited line
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(h.read())
        tf.import_graph_def(graph_def, name='')

    print('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))


initialSetup()


### Helper code for frame processing
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


############### Pistol Detection ###############

os.chdir("C:\pistol-detection\detect_pistol\Frames") #make changes to where images and csv files are stored
path = "C:\pistol-detection\detect_pistol\Frames" #same for this line
listing = os.listdir(path)
person_count = 0
with tf.Session() as sess2:
    for infile in listing:
        if not infile.endswith(".jpg"):
            continue
        image_np = cv2.imread(infile)
        
        csvfile = infile.replace(".jpg", ".csv")
        df8 = pd.read_csv(csvfile)
        wid = df8['wid'].tolist()
        hei = df8['hei'].tolist()
        head_hei = df8['head_hei'].tolist()
        px = df8['px'].tolist()
        py = df8['py'].tolist()
        pxa = df8['pxa'].tolist()
        pya = df8['pya'].tolist()
        pyha = df8['pyha'].tolist()
        
        start_time = timeit.default_timer()
        for person in range(0, 5):
            softmax_tensor = sess2.graph.get_tensor_by_name('final_result:0') # 'final_result:0'

               

                # while True:
                #             frame = grabVideoFeed()

                #             if frame is None:
                #                 raise SystemError('Issue grabbing the frame')
            halfBody = hei[person] / 3
            roi = image_np[int(py[person]):int(py[person]) + int(halfBody), int(px[person]):int(px[person]) + int(wid[person])]

            frame = cv2.resize(roi, (299, 299), interpolation=cv2.INTER_CUBIC)

                # adhere to TS graph input structure
            numpy_frame = np.asarray(frame)
            numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            numpy_final = np.expand_dims(numpy_frame, axis=0)

        
                # make prediciton
            predictions = sess2.run(softmax_tensor, {'Mul:0': numpy_final})
    
            score = predictions.item(1)
            gunScore = str(score)

 
                #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+hei[person]), (0, 0, 255), 2)
                #cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+hei[person]), (0, 255, 0), 2)

            if score > person_gun_threshold:
                cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+ (int(head_hei[person]) * 2)), (0, 0, 255), 10)
                labelBuffer = int(py[person]) - int(hei[person] * 0.1)

                    # print
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_np, gunScore, (int(px[person]), labelBuffer), font, 0.8, (0, 255, 0), 2)
                    

                cv2.imwrite('save_image/' + "frame%d.jpg" % person_count, image_np)
                print("Horizontal Angle" + str(pxa[person]) )
                print(" Vertical Angle " + str(pya[person]))
                print(" Head Vertical Angle " + str(pyha[person]))
                person_count += 1
                    
                #cv2.putText(frame, gunScore, (10, 200), font, 0.8, (0, 255, 0), 2)

            print('Took {} seconds to perform image recognition on people found'.format(timeit.default_timer() - start_time))
        cv2.imshow('frame',cv2.resize(image_np, (1024, 768)))
        if cv2.waitKey(500) & 0xFF == ord('q'):
            os.remove(location+'/'+image)
            os.remove(location+'/'+csv)
            break
        os.remove(location+'/'+image)
        os.remove(location+'/'+csv)
sess2.close()    
cv2.destroyAllWindows()    
