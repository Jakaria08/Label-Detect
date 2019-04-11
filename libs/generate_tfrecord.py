from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
import libs.dataset_util as dataset_util
from collections import namedtuple, OrderedDict
from string import punctuation

#flags = tf.app.flags
#flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#FLAGS = flags.FLAGS

id = []
name = []

def read_pbtxt():
    global id, name
    current_dir = os.getcwd()
    print(current_dir)
    new_file = os.path.join(current_dir,'Training_config','detection.pbtxt')
    print(new_file)

    infile = open(new_file,'r')
    filecontent = infile.readlines()
    infile.close()

    for line in filecontent:
        if 'id' in line:
            tmp = line.strip().split(':')
            id.append(tmp[1].strip())
        elif 'name' in line:
            tmp = line.strip().split(':')
            name.append(tmp[1].strip().strip(punctuation))
        else:
            continue

    id = [int(i) for i in id]

    print(id)
    print(name)

# TO-DO replace this with label map
def class_text_to_int(row_label):
    for ids, item in zip(id,name):
        if item == row_label:
            return ids
        else:
            None

def class_int_to_text(label):
    for ids, item in zip(id,name):
        if ids == label:
            return item
        else:
            None

def class_text_to_int_old(row_label):
    if row_label == 'empty_site':
        return 1
    elif row_label == 'single_tank':
	    return 2
    elif row_label == 'multiple_tank':
	    return 3
    else:
        None

def class_int_to_text_old(label):
    if label == 1:
        return 'empty_site'
    elif label == 2:
	    return 'single_tank'
    elif label == 3:
	    return 'multiple_tank'
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main_Tf(target_path, record_file, images_csv):
    read_pbtxt()
    print(target_path)
    writer = tf.python_io.TFRecordWriter(os.path.join(target_path, 'TFrecords', record_file))

    examples = pd.read_csv(os.path.join(target_path, 'TFrecords', images_csv))
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, target_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), 'TFrecords')
    print('Successfully created the TFRecords: {}'.format(os.path.join(target_path, 'TFrecords')))

def test():
    print('Working...')


#if __name__ == '__main__':
#    tf.app.run()
