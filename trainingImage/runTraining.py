import os
from pathlib import Path

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

class Trainer:

    def __init__(self,tf_training, pbtxt_file, model_train, config_file):

        self.Tf_training = tf_training
        self.pbtxt_training = pbtxt_file
        self.pre_model_training = model_train
        self.config_file = config_file

        print(self.Tf_training)
        print(self.pbtxt_training)
        print(self.pre_model_training)
        print(self.config_file)

        os.chdir('..')
        print(os.getcwd())

        progressbar = ProgressBar(100, title = "Training Started...")
        progressbar.setValue(2)

        training_command = """python train.py
                              --logtostderr
                              --train_dir=training_FRCNN_resnet101_coco/
                              --pipeline_config_path=training_FRCNN_resnet101_coco/faster_rcnn_resnet101_coco.config"""
        #os.system(training_command)
        frozen_graph_command = """python export_inference_graph.py
                                  --input_type image_tensor
                                  --pipeline_config_path /home/hipstudents/tensorflow/models/research/object_detection/training_FRCNN_resnet101_coco/faster_rcnn_resnet101_coco.config
                                  --trained_checkpoint_prefix /home/hipstudents/tensorflow/models/research/object_detection/training_FRCNN_resnet101_coco/model.ckpt-63646
                                  --output_directory /home/hipstudents/tensorflow/models/research/object_detection/training_FRCNN_resnet101_coco/"""
        os.system(frozen_graph_command)
