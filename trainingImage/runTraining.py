import os
import glob
import in_place
from pathlib import Path
from sys import platform as _platform

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

        head, tail = os.path.split(self.Tf_training)
        test_path = os.path.join(head, 'test.record')

        head1, tail1 = os.path.split(self.pre_model_training)
        new_model_path = os.path.join(head1,'model.ckpt')
        """
        with in_place.InPlace(self.config_file) as file:
            for line in file:
                if 'PATH_TO_BE_CONFIGURED_MODEL' in line:
                    line = line.replace('PATH_TO_BE_CONFIGURED_MODEL', new_model_path)
                elif 'PATH_TO_BE_CONFIGURED_TRAIN' in line:
                    line = line.replace('PATH_TO_BE_CONFIGURED_TRAIN', self.Tf_training)
                elif 'PATH_TO_BE_CONFIGURED_TEST' in line:
                    line = line.replace('PATH_TO_BE_CONFIGURED_TEST', test_path)
                else:
                    line = line.replace('PATH_TO_BE_CONFIGURED_PBTXT', self.pbtxt_training)
                file.write(line)
        """
        train_dir = os.path.join(head,'Training_Folder')

        if os.path.exists(train_dir):
            print('Directory Exists!')
        else:
            os.mkdir(train_dir)

        current_dir = os.getcwd()
        top_dir_wind = os.path.join(*(current_dir.split(os.path.sep)[1:2]))
        top_dir_win = os.path.join('/',top_dir_wind)
        top_dir_linuxx = os.path.join(*(current_dir.split(os.path.sep)[1:3]))
        top_dir_linux = os.path.join('/',top_dir_linuxx)

        if _platform == "linux" or _platform == "linux2":
            print('linux')
            path = os.path.join(top_dir_linux,'tensorflow/models/research/object_detection')
            print(path)
        elif _platform == "win64" or "win32":
            print('windows')
            path = os.path.join(top_dir_win,'tensorflow/models/research/object_detection')
        else:
            print('Not supported!')

        os.chdir(path)

        progressbar = ProgressBar(100, title = "Training Started...")
        #progressbar.setValue(2)

        training_command = "python train.py --logtostderr --train_dir="+train_dir+" --pipeline_config_path="+self.config_file
        #os.system(training_command)

        get_strings = []
        checkpoint_path = train_dir

        for checkpoint in glob.glob(checkpoint_path+'/*.index'):
            get_strings.append(int(checkpoint.split('.')[1].split('-')[1]))

        model_number = max(get_strings)
        model_name = 'model.ckpt-'+str(model_number)
        model_checkpoint_path = os.path.join(train_dir,model_number)
        print(model_checkpoint_path)

        frozen_graph_command = """python export_inference_graph.py \
                                  --input_type image_tensor \
                                  --pipeline_config_path """+self.config_file+""" \
                                  --trained_checkpoint_prefix """+model_name+""" \
                                  --output_directory """+train_dir
        os.system(frozen_graph_command)
