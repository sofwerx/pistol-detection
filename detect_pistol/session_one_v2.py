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

#!!!!!!!!!!!!!!!!!!!!!!!!!! Make changes to lines: 44, 88, 98, 122, 123

# Science Thresholds

person_threshold = 0.50
person_gun_threshold = 0.80


# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sys.path.append("..")

os.chdir("/models-master/research/object_detection/") #change to where object_detection folder is
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
MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08' #should be in object detection folder


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
label_lines = [line.rstrip() for line in tf.gfile.GFile("C:/pistol-detection/tf_files/retrained_labels.txt")] #edited line. Change path


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    #with tf.gfile.FastGFile('/tf_files/retrained_graph.pb', 'rb') as h: #original line
    with tf.gfile.FastGFile('C:/pistol-detection/tf_files/retrained_graph.pb', 'rb') as h: #edited line. Change path
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


################# Person Object Detection #############################
#url = "rtsp://admin:1qazxsw2!QAZXSW@@datascience.opswerx.org:20044"
url = "C:/FrameData/ReceptionWestVideo.mp4"
cap = cv2.VideoCapture(url)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

os.chdir("C:\pistol-detection\detect_pistol\Frames") #path to directory where images and csv files will be stored
path = "C:\pistol-detection\detect_pistol\Frames" #path to directory where images will be stored
count = 0
nameCount = 0
waitTime = 1

with tf.Session(graph=detection_graph) as sess:
    while(cap.isOpened()):
        ret, image_np = cap.read()
        if not ret:
            cap = cv2.VideoCapture(url)
            continue
        count+=1
        if not count%27 == 0:
            continue
        height, width, channels = image_np.shape
        out.write(image_np) #records video
        
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
#             vis_util.visualize_boxes_and_labels_on_image_array(
#               image_np,
#               np.squeeze(boxes),
#               np.squeeze(classes).astype(np.int32),
#               np.squeeze(scores),
#               category_index,
#               use_normalized_coordinates=True,
#               line_thickness=8)


        xfov = 71
        yfov = 44  
        ch = 180
        imageHeight = int(height)
        imageWidth = int(width)
        imageHeightCenter = imageHeight / 2
        imageWidthCenter = imageWidth / 2
        pixelDegreeHorizontal = float(xfov) / imageWidth
        pixelDegreeVertical = float(yfov) / imageHeight

            # Convert tensorflow data to pandas data frames
            # print boxes
        df = pd.DataFrame(boxes.reshape(300, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
        df1 = pd.DataFrame(classes.reshape(300, 1), columns=['classes'])
        df2 = pd.DataFrame(scores.reshape(300, 1), columns=['scores'])
        df5 = pd.concat([df, df1, df2], axis=1)

            # Transform box bound coordinates to pixel coordintate
        df5['y_min_t'] = df5['y_min'].apply(lambda x: x * imageHeight)
        df5['x_min_t'] = df5['x_min'].apply(lambda x: x * imageWidth)
        df5['y_max_t'] = df5['y_max'].apply(lambda x: x * imageHeight)
        df5['x_max_t'] = df5['x_max'].apply(lambda x: x * imageWidth)

            # Create objects pixel location x and y
            # X
        df5['ob_wid_x'] = df5['x_max_t'] - df5["x_min_t"]
        df5['ob_mid_x'] = df5['ob_wid_x'] / 2
        df5['x_loc'] = df5["x_min_t"] + df5['ob_mid_x']
            # Y
        df5['ob_hgt_y'] = df5['y_max_t'] - df5["y_min_t"]
        df5['ob_mid_y'] = df5['ob_hgt_y'] / 2
        df5['y_loc'] = df5["y_min_t"] + df5['ob_mid_y']
            # Head
        df5['ob_head_y'] = df5['ob_hgt_y'] / 7.5 / 2
            
        df5['y_head_loc'] = df5["y_min_t"] + df5['ob_head_y']
  

            # Find object degree of angle, data is sorted by score, select person with highest score
        df5['object_x_angle'] = df5['x_loc'].apply(lambda x: -(imageWidthCenter - x) * pixelDegreeHorizontal)
        df5['object_y_angle'] = df5['y_loc'].apply(lambda x: (imageHeightCenter - x) * pixelDegreeVertical)
        df5['head_y_angle'] = df5['y_head_loc'].apply(lambda x: (imageHeightCenter - x) * pixelDegreeVertical)

        df6 = df5.loc[(df5['classes'] == 1 ) %  (df5['scores'] > person_threshold)]
            #df6 = df6.loc[df6['scores'] > 0.80]

        if (df6.empty) or (df6.iloc[0]['scores'] < person_threshold):
            continue


        for i in range(0,len(df6.index)):

    

            w = int(df6.iloc[i]['ob_wid_x'])
            x = int(df6.iloc[i]['x_min_t'])
            h = int(df6.iloc[i]['ob_hgt_y'])
            y = int(df6.iloc[i]["y_min_t"])

            HAOB = df6.iloc[i]['object_x_angle'] 
            HAOB_str = str(round(HAOB, 4))
                
            VAOB = df6.iloc[i]['object_y_angle']
            VAOB_str = str(round(VAOB, 4))
                
            head_AOB = df6.iloc[i]['head_y_angle']
            head_size = df6.iloc[i]['ob_head_y']
            head_y  = df6.iloc[i]['y_head_loc']
#                 print y,h,VAOB
#                 print head_y, head_size,head_AOB
   

                #labelBuffer = int(df6.iloc[0]['y_min_t']) - int(df6.iloc[0]['ob_hgt_y'] * 0.1)

                # print
            font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(image_np, AOB_str, (int(df6.iloc[0]['x_min_t']), labelBuffer), font, 0.8, (0, 255, 0), 2)

            halfBody = h / 3
            bigBody1 = w * 3
            bigBody2 = x - (w * 2)
                #roi = image_np[y:y + halfBody, x:x + w]
                #cv2.rectangle(image_np, (x,y), (x+w, y+halfBody), (0, 255, 0), 2)
                #cv2.imwrite('save_image/' + "frame%d.jpg" % count, roi)


            wid.append(w)
            hei.append(h)
            head_hei.append(head_size)
            px.append(x)
            py.append(y)
            pxa.append(HAOB)
            pya.append(VAOB)
            pyha.append(head_AOB)
                #print boxes
        metadata = []
        metadata.append(wid)
        metadata.append(hei)
        metadata.append(head_hei)
        metadata.append(px)
        metadata.append(py)
        metadata.append(pxa)
        metadata.append(pya)
        metadata.append(pyha)
        df7 = pd.DataFrame(metadata)
        df7 = df7.transpose()
        df7.columns = ['wid', 'hei', 'head_hei', 'px', 'py', 'pxa', 'pya', 'pyha']
        
        name = "rec_frame"+str(nameCount)+".jpg"
        cv2.imwrite(os.path.join(path,name), image_np)
        nameCount+=1
        csvfile = name.replace(".jpg", ".csv")
        df7.to_csv(csvfile)
        cv2.imshow('frame', image_np)
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
sess.close()
cap.release()
out.release()
cv2.destroyAllWindows()