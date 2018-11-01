# coding: utf-8
########################################################################################
#### Ctrl-f '#!!!' to find lines that may need to be changed, depending on host.    ####
#### As is, this script runs on the container built by the Dockerfile in this repo. ####
########################################################################################
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
import sys
import numpy as np

from slackclient import SlackClient
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as img
from IPython.display import Image, display, clear_output
from elasticsearch import Elasticsearch
from datetime import datetime
from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists)


# Hold warnings
import warnings
warnings.filterwarnings('ignore')



#######################################################################
################ Initialize Functions/Variables #######################
#######################################################################
# Global Variables

# Camera
RECEPTION_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20043"
RECEPTION_WEST = "rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20044"
DIRTYWERX_NORTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20045"
DIRTYWERX_SOUTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20046"
THUNDERDRONE_INDOOR_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20047"
THUNDERDRONE_INDOOR_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20048"
OUTSIDE_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20049"
OUTSIDE_NORTH_WEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20050"
OUTSIDE_NORTH="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20051"
OUTSIDE_NORTH_EAST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20052"
DIRTYWERX_RAMP="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20053"
TEST="rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20043" #!!!

def notify(channelToken, isTest=true):
    """Sends a message to specified channel
    Channel values are the last 9 characters in the URL for a channel
    Example channel: C0JJACWSX (#general channel)
    DO NOT put the Slack token on Github or anywhere public. Instead, 
    define the 75-character token as an environment variable using
    `export SLACK_API_TOKEN=[insert slack token here, no brackets]`"""
    slack_token = os.environ["SLACK_API_TOKEN"] 
	sc = SlackClient(slack_token)
	if isTest:
		msgPrefix = "THIS IS A TEST"
	else:
		msgPrefix = "***WARNING***"
	sc.api_call(
	  "chat.postMessage",
	  channel=channelToken,
	  text="{}: A person with a gun has been detected on the premesis. The doors are being locked now.".format(msgPrefix)
	)

# Setup ES 
try:
    es = Elasticsearch(
        [
            'https://elastic:diatonouscoggedkittlepins@elasticsearch.orange.opswerx.org:443'
        ],
        verify_certs=True
    )
    print("ES - Connected.")
except Exception as ex:
    print("Error: ", ex)


# GPU Percentage
#gpuAmount = int((sys.argv)[2]) * 0.1 #!!!


# Camera Selection
url = globals()[str((sys.argv)[1])] #!!!
#url=TEST
print(url)


# Science Thresholds
person_threshold = 0.50
person_gun_threshold = 0.60


# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = gpuAmount #!!!
session = tf.Session(config=config)

os.chdir("/tensorflow/models/research/object_detection/") #!!!

# Get Video and dimensions
cap = cv2.VideoCapture(url) #!!!
print(cap)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
os.chdir("/tensorflow/models/research/object_detection/") #!!!

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
MODEL_NAME ='faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
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
label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")] #!!!


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    with tf.gfile.FastGFile('/tf_files/retrained_graph.pb', 'rb') as h: #!!!
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(h.read())
        tf.import_graph_def(graph_def, name='')

    print('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))


initialSetup()


# ## Helper code for frame processing
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

minioClient = Minio('minio.supermicro3.opswerx.org',
                  access_key='admin',
                  secret_key='Doolittle123',
                  secure=True)

#######################################################################
############# Perform object Detection and Recognition ################
#######################################################################



################# Person Object Detection #############################

count = 0
nameCount = 0
waitTime = 1
path = "\pistol-detection\detect_pistol\Frames" #!!! Directory where frames and data will be stored
os.chdir(path)

####################Video Variables##########################


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print ("fps " + str(fps))

################################################################33

# Loop Frame by Frame
with tf.Session(graph=detection_graph) as sess:
    while(cap.isOpened()):
        ret, image_np = cap.read()
        if not ret:
            cap = cv2.VideoCapture(url)
            print("Error capturing frames")
            continue
        count+=1
        if not count%27 == 0:
            continue
        
        #make image brighter
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v += 255
        final_hsv = cv2.merge((h, s, v))
        image_np = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        start_time = timeit.default_timer()
        
        # Loop Variables
        wid = []
        hei = []
        head_hei = []
        px  = []
        py =  []
        pxa = []
        pya = []
        pyha = []
        
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        print('Took {} seconds to find people in image'.format(timeit.default_timer() - start_time))
        # print scores, classes
                
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)


                # Convert tensorflow data to pandas data frams
                # print boxes
        df = pd.DataFrame(boxes.reshape(300, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
        df1 = pd.DataFrame(classes.reshape(300, 1), columns=['classes'])
        df2 = pd.DataFrame(scores.reshape(300, 1), columns=['scores'])
        df4 = pd.concat([df, df1, df2], axis=1)
        df5 = df4[(df4.classes == 1) & (df4.scores > person_threshold)]
                #df5 = df4.loc[(df4['classes'] == 1 ) &  (df4['scores'] > person_threshold)]
                #print df5

                # Transform box bound coordinates to pixel coordintate

        df5['y_min_t'] = df5['y_min'].apply(lambda x: x * height)
        df5['x_min_t'] = df5['x_min'].apply(lambda x: x * width)
        df5['y_max_t'] = df5['y_max'].apply(lambda x: x * height)
        df5['x_max_t'] = df5['x_max'].apply(lambda x: x * width)

                # Create objects pixel location x and y
                # X
        df5['ob_wid_x'] = df5['x_max_t'] - df5["x_min_t"]
        df5['ob_mid_x'] = df5['ob_wid_x'] / 2
        df5['x_loc'] = df5["x_min_t"] + df5['ob_mid_x']
                # Y
        df5['ob_hgt_y'] = df5['y_max_t'] - df5["y_min_t"]
        df5['ob_mid_y'] = df5['ob_hgt_y'] / 2
        df5['y_loc'] = df5["y_min_t"] + df5['ob_mid_y']

        df6 = df5


                # Scan People in Frame

        if (df6.empty) or (df6.iloc[0]['scores'] < person_threshold):
            continue


        for i in range(0,len(df6.index)):


            w = int(df6.iloc[i]['ob_wid_x'])
            x = int(df6.iloc[i]['x_min_t'])
            h = int(df6.iloc[i]['ob_hgt_y'])
            y = int(df6.iloc[i]["y_min_t"])

            wid.append(w)
            hei.append(h)
            px.append(x)
            py.append(y)
        
        #place relevant data in a DataFrame
        print('Took {} seconds to gather data'.format(timeit.default_timer() - start_time))
        metadata = []
        metadata.extend((wid, hei, px, py))
        df7 = pd.DataFrame(metadata)
        df7 = df7.transpose()
        df7.columns = ['wid', 'hei', 'px', 'py']
        
        #store current frame and data about frame in a directory (directory location determined by line 212)
        name = "rec_frame"+str(nameCount)+".jpg"
        
        cv2.imwrite(os.path.join(path,name), image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) #lowers image resolution and saves image
        nameCount+=1
        csvfile = name.replace(".jpg", ".csv")
        df7.to_csv(csvfile)
        
        #send frame and data to SOFWERX minio client
        try:
            minioClient.fput_object('person-camera', name, name)
            minioClient.fput_object('person-camera', csvfile, csvfile)
        except ResponseError as err:
            print(err)
        
        #remove the frame and data from the directory (they are stored in the minio client and don't need to take up space in the directory)
        os.remove(name)
        os.remove(csvfile)
        
        print('Took {} seconds to reach end of session'.format(timeit.default_timer() - start_time))
sess.close()
print('Took {} seconds to find people in image'.format(timeit.default_timer() - start_time))

