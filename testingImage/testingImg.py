from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

from tensorflow.python.client import timeline
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    #needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

class ProgressBar(QProgressDialog):
    def __init__(self, max, title):
        super().__init__()
        self.setMinimumDuration(0)
        self.setWindowTitle(title)
        self.setModal(True)

        self.setValue(0)
        self.setMinimum(0)
        self.setMaximum(max)
        self.setCancelButton(None)
        self.setLabelText("Work in progress, Please Wait..")

        self.show()

class Testing:

    def __init__(self, test_image_path, test_output_name,
                 test_output_dir, test_height, test_width,
                 test_model_path):
        self.testImagePath = test_image_path
        self.testOutputName = test_output_name
        self.testOutputDir = test_output_dir
        self.im_height = test_height
        self.im_width = test_width
        self.PATH_TO_FROZEN_GRAPH = test_model_path
        self.TEST_IMAGES_DIR = test_output_dir
        self.outpath = list()

        print(f"test image path: {self.testImagePath}")
        print(f"Output name: {self.testOutputName}")
        print(f"Output dir: {self.testOutputDir}")
        print(f"Image height: {self.im_height}")
        print(f"Image Width: {self.im_width}")
        print(f"Model path: {self.PATH_TO_FROZEN_GRAPH}")
        print(f"Output dir: {self.TEST_IMAGES_DIR}")

        self.slice_test_image(self.testImagePath, self.testOutputName,
                              self.testOutputDir, self.im_height,
                              self.im_width)


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    def slice_test_image(self, test_image_path, out_name, out_dir, sliceHeight, sliceWidth,
             zero_frac_thresh=0.2, overlap=0.2, slice_sep='-',
             out_ext='.png', verbose=False):

        '''Slice large satellite image into smaller pieces,
        ignore slices with a percentage null greater then zero_frac_thresh
        Assume three bands.
        if out_ext == '', use input file extension for saving'''
        print(test_image_path)
        if os.path.exists(out_dir):
            print('Directory Exists!')
            return

        os.mkdir(out_dir)

        image0 = cv2.imread(test_image_path, 1)  # color
        if len(out_ext) == 0:
            ext = '.' + image_path.split('.')[-1]
        else:
            ext = out_ext

        win_h, win_w = image0.shape[:2]
        print(win_h,win_w)

        # if slice sizes are large than image, pad the edges
        pad = 0
        if sliceHeight > win_h:
            pad = sliceHeight - win_h
        if sliceWidth > win_w:
            pad = max(pad, sliceWidth - win_w)
        # pad the edge of the image with black pixels
        if pad > 0:
            border_color = (0,0,0)
            image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=border_color)

        win_size = sliceHeight*sliceWidth

        t0 = time.time()
        n_ims = 0
        n_ims_for_progress = 0
        n_ims_nonull = 0
        dx = int((1. - overlap) * sliceWidth)
        dy = int((1. - overlap) * sliceHeight)

        for y0 in range(0, image0.shape[0], dy):#sliceHeight):
            for x0 in range(0, image0.shape[1], dx):#sliceWidth):
                n_ims_for_progress += 1

        progressbar = ProgressBar(n_ims_for_progress, title = "Slicing Images...")
        print("Where is the error??????")
        for y0 in range(0, image0.shape[0], dy):#sliceHeight):
            for x0 in range(0, image0.shape[1], dx):#sliceWidth):
                n_ims += 1
                print(f"within the main loop {n_ims}")

                #time.sleep(0.1)
                progressbar.setValue(n_ims)

                if (n_ims % 100) == 0:
                    print (n_ims)

                # make sure we don't have a tiny image on the edge
                if y0+sliceHeight > image0.shape[0]:
                    y = image0.shape[0] - sliceHeight
                else:
                    y = y0
                if x0+sliceWidth > image0.shape[1]:
                    x = image0.shape[1] - sliceWidth
                else:
                    x = x0
                print(f"within the main loop before extraction {n_ims}")
                # extract image
                window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                # get black and white image
                window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

                # find threshold that's not black
                # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
                ret,thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                non_zero_counts = cv2.countNonZero(thresh1)
                zero_counts = win_size - non_zero_counts
                zero_frac = float(zero_counts) / win_size
                #print "zero_frac", zero_fra
                # skip if image is mostly empty
                print(f"within the main loop before if {n_ims}")
                if zero_frac >= zero_frac_thresh:
                    if verbose:
                        print ("Zero frac too high at:", zero_frac)
                    continue
                # else save
                else:
                    #self.outpath = os.path.join(outdir, out_name + \
                    #'|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                    #'_' + str(pad) + ext)
                    outpaths = os.path.join(out_dir, out_name + \
                    slice_sep + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                    '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext)

                    self.outpath.append(os.path.join(out_dir, out_name + \
                    slice_sep + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                    '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext))

                #self.outpath = os.path.join(outdir, 'slice_' + out_name + \
                #'_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                #'_' + str(pad) + '.jpg')

                    if verbose:
                        print ("outpaths:", outpaths)
                    print(f"within the main loop before img write{n_ims}")
                    cv2.imwrite(outpaths, window_c)
                    n_ims_nonull += 1

        print(self.outpath[0])
        print ("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull, \
                "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
        print ("Time to slice", test_image_path, time.time()-t0, "seconds")

        self.start_detecting()

    def start_detecting(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                start_time = time.time()
                new_global_boxes = [list() for f in range(len(self.outpath))]
                new_global_boxes_sup = list()

                path_size = len(self.outpath)
                progressbar = ProgressBar(path_size, title = "Detecting...")
                progressbar.setValue(0)

                for i in range(len(self.outpath)):
                    image = Image.open(os.path.join(self.TEST_IMAGES_DIR, self.outpath[i]))
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = self.load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    (boxes, scores, classes, num) = sess.run(\
                    [detection_boxes, detection_scores, detection_classes, num_detections], \
                    feed_dict={image_tensor: image_np_expanded}, \
                    options=options, run_metadata=run_metadata)

                    if i%10 == 0:
                        print(i)

                    boxs = np.squeeze(boxes)
                    scors = np.squeeze(scores)
                    #finding global co-ordinate
                    outp = os.path.basename(self.outpath[i])
                    box_co = outp.split('_')
                    length = len(box_co)

                    x_co = float(box_co[length - 6])
                    y_co = float(box_co[length - 7].split('-')[1])

                    for j in range(boxs.shape[0]):
                        if scors is None or scors[j] > 0.5:
                            box = tuple(boxs[j].tolist())

                            ymin, xmin, ymax, xmax = box

                            ymin = y_co + (ymin*self.im_height)
                            xmin = x_co + (xmin*self.im_width)
                            ymax = y_co + (ymax*self.im_height)
                            xmax = x_co + (xmax*self.im_width)

                            #new_global_boxes[i].append([ymin, xmin, ymax, xmax])
                            new_global_boxes_sup.append([ymin, xmin, ymax, xmax])

                    #time.sleep(0.1)
                    progressbar.setValue(i)

                print('Total testing time after evaluating %d images : %.3f sec'%(i, time.time()-start_time))

                all_boxes = np.array(new_global_boxes_sup)
                print(all_boxes.shape)
                self.save_box(all_boxes)

    # Malisiewicz et al.
    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        y1 = boxes[:,0]
        x1 = boxes[:,1]
        y2 = boxes[:,2]
        x2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            #print(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

            # return only the bounding boxes that were picked using the
            # integer data type
        return boxes[pick].astype("int")

    def save_box(self, all_boxes):
        print(self.testOutputDir)
        test_out_directory = os.path.join(self.testOutputDir, "boxCSV")
        if os.path.exists(test_out_directory):
            print('Directory Exists!')
            return
        os.mkdir(test_out_directory)
        np.savetxt(os.path.join(test_out_directory, "all_boxes.csv"), all_boxes, delimiter=",")
        np.savetxt(os.path.join(test_out_directory, "all_boxes_dec.csv"), all_boxes, fmt="%d", delimiter=",")
        single_boxes = self.non_max_suppression_fast(all_boxes, 0.1)
        print(single_boxes[10])
        np.savetxt(os.path.join(test_out_directory, "single_boxes.csv"), single_boxes, fmt="%d", delimiter=",")
