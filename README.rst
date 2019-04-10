Label-Detection
======================

Label-Detection  is a graphical image annotation tool and using this tool user can also train and test large satellite images. User can create small patches from large image, annotate it, create training and testing data, select model, train-test the model. The annotations (labeling) part of this application is based
on `this repository <https://github.com/tzutalin/labelImg>`__.

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
by `ImageNet <http://www.image-net.org/>`__.  

User can use many deep learning models such as Faster RCNN Resnet or SSD Mobilenet.

We can see a example of image labeling and detection in the following images. Sample Labeling, Traing and Testing procedures can be found in the videos that are posted below the images.

.. image:: https://user-images.githubusercontent.com/7825643/55756403-af622e80-5a0e-11e9-81fd-873b54cae6d9.png
     :alt: Demo Image
.. image:: https://user-images.githubusercontent.com/7825643/55766217-e1848800-5a30-11e9-808d-dcfbf64ff387.png
     :alt: Demo Image

`Watch a demo video <https://youtu.be/FFe5Y7u7APs>`__ [Labeling] [Created by tzutalin]

`Watch a demo video <https://youtu.be/WNz9Djt9ETc>`__ [Training Part-1]

`Watch a demo video <https://youtu.be/nbvI0EviPbI>`__ [Training Part-2]

`Watch a demo video <https://youtu.be/VCEd9WKQpWA>`__ [Testing]

Installation
------------------

Build from source
~~~~~~~~~~~~~~~~~

Linux/Ubuntu requires at least `Python
3.6 <https://www.python.org/getit/>`__ and has been tested with `PyQt
5.8 <https://www.riverbankcomputing.com/software/pyqt/intro>`__.

Ubuntu Linux
^^^^^^^^^^^^

`Install Tensorflow with GPU support <https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25>`_

`Install Tensorflow with GPU support <https://www.tensorflow.org/install/gpu>`_ [Tensorflow Documentation]

.. code::

    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    

Windows + Anaconda
^^^^^^^

Download and install `Anaconda <https://www.anaconda.com/download/#download>`__ (Python 3.6+)

`Install Tensorflow with GPU support <https://www.anaconda.com/tensorflow-in-anaconda/>`_

`Hardware Requirements <https://www.tensorflow.org/install/gpu#windows_setup>`_

Open the Anaconda Prompt and go to the `labelImg <#labelimg>`__ directory

.. code::

    conda install pip
    pip install -r requirements/requirements-linux-python3.txt
    pyrcc5 -o resources.py resources.qrc
    python labelImg.py

Tensorflow Object Detection API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Tensoflow object detection api is used only for training purpose. By using this api we can use different types of models to train. I am also integrating models that are not in the api.
2. Ubuntu: Go to the '/home/your-username/' directory and create a folder named 'tensorflow' and go into the folder. Now download or clone `this repository <https://github.com/tensorflow/models>`_ and follow `this installation process. <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md>`_.
3. Windows: Download `Git Bash, <https://github.com/git-for-windows/git/releases/download/v2.21.0.windows.1/Git-2.21.0-64-bit.exe>`_ go to the 'C:/Users/your-username/' directory of your computer, create a folder named 'tensorflow' and go into the folder. Now follow this `tutorial <https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b>`_ or `this tutorial <https://basecodeit.com/blog/installing-tensorflow-with-object-detection-api-part-1/>`_

Usage
-----
Annotation
----------
Steps (PascalVOC)
~~~~~

1. Build and launch using the instructions above.
2. Click 'Change default saved annotation folder' in Menu/File
3. Click 'Open Dir'
4. Click 'Create RectBox'
5. Click and release left mouse to select a region to annotate the rect
   box
6. You can use right mouse to drag the rect box to copy or move it

The annotation will be saved to the folder you specify.

You can refer to the below hotkeys to speed up your workflow.

Training
----------
Steps 
~~~~~

1. Select 'File -> Open Image and Slice' [Ctrl+i] 
2. Select the desired Satellite Image and then can enter the slice/patch height and width. The default value is 512 pixels.
3. Then select 'Start Slicing'
4. After Slicing the big image, you can see a new directory on the image's directory and within it you can see image slices/patches.
5. Annotate the images and save the .xml files according to the 'Annotation' section discussed above.
6. Select 'File -> Select Directory to Create TFrecords' [Ctrl+t] and select the directory that contains all the .xml files.
7. Then TFRecords files for training and testing will be created under TFrecords folder withing the directory selected in step 6.
8. Select 'Start Training' [Ctrl+Shift+t] 
9. Select the TFRecord file for training which is 'train.record' 
10. Select 'detection.pbtxt' and a config file from 'Label-Detect/Training_config' directory. If you want to use Faster R-CNN ResNet-101 then select the corresponding file otherwise you can select the config file for SSD MobileNet.
11. Downdload the `Faster R-CNN Resnet-101 model,  <http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz>`_ extract it and select the 'model.ckpt.index' file for the model file.  

Testing
----------
Steps 
~~~~~

1.


Hotkeys
~~~~~~~

+--------------------+--------------------------------------------+
| Ctrl + u           | Load all of the images from a directory    |
+--------------------+--------------------------------------------+
| Ctrl + r           | Change the default annotation target dir   |
+--------------------+--------------------------------------------+
| Ctrl + s           | Save                                       |
+--------------------+--------------------------------------------+
| Ctrl + d           | Copy the current label and rect box        |
+--------------------+--------------------------------------------+
| Ctrl + i           | Open Image and Slice                       |
+--------------------+--------------------------------------------+
| Ctrl + t           | Select Directory to Create TFrecords       |
+--------------------+--------------------------------------------+
| Ctrl + Shift + t   | Start Training                             |
+--------------------+--------------------------------------------+
| Ctrl + Shift + w   | Load Test Image to Get the Results         |
+--------------------+--------------------------------------------+
+--------------------+--------------------------------------------+
| Space              | Flag the current image as verified         |
+--------------------+--------------------------------------------+
| w                  | Create a rect box                          |
+--------------------+--------------------------------------------+
| d                  | Next image                                 |
+--------------------+--------------------------------------------+
| a                  | Previous image                             |
+--------------------+--------------------------------------------+
| del                | Delete the selected rect box               |
+--------------------+--------------------------------------------+
| Ctrl++             | Zoom in                                    |
+--------------------+--------------------------------------------+
| Ctrl--             | Zoom out                                   |
+--------------------+--------------------------------------------+
| ↑→↓←               | Keyboard arrows to move selected rect box  |
+--------------------+--------------------------------------------+

**Verify Image:**

When pressing space, the user can flag the image as verified, a green background will appear.
This is used when creating a dataset automatically, the user can then through all the pictures and flag them instead of annotate them.

How to contribute
~~~~~~~~~~~~~~~~~

Send a pull request

License
~~~~~~~
`Free software: MIT license <https://github.com/tzutalin/labelImg/blob/master/LICENSE>`_

Citation
~~~~~~~~
Tzutalin. LabelImg. Git code (2015). https://github.com/tzutalin/labelImg

Related
~~~~~~~
`App Icon based on Icon by Nick Roach (GPL)` <https://www.elegantthemes.com/> <https://www.iconfinder.com/icons/1054978/shop_tag_icon> __

