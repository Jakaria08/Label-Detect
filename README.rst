Label-Detection
======================

.. image:: https://img.shields.io/pypi/v/labelimg.svg
        :target: https://pypi.python.org/pypi/labelimg

.. image:: https://img.shields.io/travis/tzutalin/labelImg.svg
        :target: https://travis-ci.org/tzutalin/labelImg

LabelImg-Detection  is a graphical image annotation tool and using this tool user can also train and test large satellite images. User can create small patches from large image, annotate it, create training and testing data, select model, train-test the model.

It is written in Python and uses Qt for its graphical interface.

Annotations are saved as XML files in PASCAL VOC format, the format used
by `ImageNet <http://www.image-net.org/>`__.  Besdies, it also supports YOLO format

User can use many deep learning models such as Faster RCNN Resnet or SSD Mobilenet.

We can see a example of image labeling and detection in the following images. Sample Labeling, Traing and Testing procedures can be found in the videos that are posted below the images.

.. image:: https://user-images.githubusercontent.com/7825643/55756403-af622e80-5a0e-11e9-81fd-873b54cae6d9.png
     :alt: Demo Image
.. image:: https://user-images.githubusercontent.com/7825643/55766217-e1848800-5a30-11e9-808d-dcfbf64ff387.png
     :alt: Demo Image

`Watch a demo video <https://youtu.be/FFe5Y7u7APs>`__ [Labeling]

`Watch a demo video <https://youtu.be/WNz9Djt9ETc>`__ [Training]

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

.. code::

    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    

`Install Tensorflow with GPU support <https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25>`_

Windows + Anaconda
^^^^^^^

Download and install `Anaconda <https://www.anaconda.com/download/#download>`__ (Python 3.6+)

Open the Anaconda Prompt and go to the `labelImg <#labelimg>`__ directory

.. code::

    conda install pyqt=5
    pyrcc5 -o resources.py resources.qrc
    python labelImg.py


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

Steps (YOLO)
~~~~~

1. In ``data/predefined_classes.txt`` define the list of classes that will be used for your training.

2. Build and launch using the instructions above.

3. Right below "Save" button in toolbar, click "PascalVOC" button to switch to YOLO format.

4. You may use Open/OpenDIR to process single or multiple images. When finished with single image, click save.

A txt file of yolo format will be saved in the same folder as your image with same name. A file named "classes.txt" is saved to that folder too. "classes.txt" defines the list of class names that your yolo label refers to.

Note:

- Your label list shall not change in the middle of processing a list of images. When you save a image, classes.txt will also get updated, while previous annotations will not be updated.

- You shouldn't use "default class" function when saving to YOLO format, it will not be referred.

- When saving as YOLO format, "difficult" flag is discarded.

Create pre-defined classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can edit the
`data/predefined\_classes.txt <https://github.com/tzutalin/labelImg/blob/master/data/predefined_classes.txt>`__
to load pre-defined classes

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

