import os
from pathlib import Path

class Trainer:

    def __init__(self,tf_training, pbtxt_file, model_train):

        self.Tf_training = tf_training
        self.pbtxt_training = pbtxt_file
        self.pre_model_training = model_train

        print(self.Tf_training)
        print(self.pbtxt_training)
        print(self.pre_model_training)

        p = Path(__file__).parents[2]
        print(p)
        final_p = os.path.join(p,train.py)

        training_command = 'python '+final_p+' --logtostderr --train_dir=training_FRCNN_resnet101_coco/ --pipeline_config_path=training_FRCNN_resnet101_coco/faster_rcnn_resnet101_coco.config'
        os.system(training_command)
