"""
Example usage:

    python create_wider_tf_record.py \
        --train_data_dir=./WIDER_train/images \
        --val_data_dir=./WIDER_val/images \
        --train_examples_path=./examples_files/train_examples.txt \
        --val_examples_path=./examples_files/val_examples.txt \
        --train_anno_path=./wider_face_split/wider_face_train_bbx_gt.txt \
        --val_anno_path=./wider_face_split/wider_face_val_bbx_gt.txt \
        --label_map_path=./wider_label_map.pbtxt \
        --output_dir=./tfrecord_data \
        --num_shards=10
"""

import contextlib2
import hashlib
import io
import logging
import os
from utils import tf_record_creation_util
from utils import dataset_util
from utils import label_map_util
from PIL import Image
import random
import read_annotations as ra
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('train_data_dir', '', 'Training data directory')
flags.DEFINE_string('val_data_dir', '', 'Valuation data directory')
flags.DEFINE_string('train_examples_path', '', 'Training examples file path')
flags.DEFINE_string('val_examples_path', '', 'Valuation examples file path')
flags.DEFINE_string('train_anno_path', '', 'Training annotation file path')
flags.DEFINE_string('val_anno_path', '', 'Valuation annotation file path')
flags.DEFINE_string('label_map_path', '', 'Lable map file path')
flags.DEFINE_string('output_dir', '', 'Output directory for the TFRecord files')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
FLAGS = flags.FLAGS


def dict_to_tf_example(anno,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
    """
    Transform dict to TF example
    :param anno: image annotation
    :param label_map_dict: label map for each image
    :param image_subdirectory: image path directory
    :param ignore_difficult_instances: whether ignore difficult image samples
    :return: tf_example
    """
    img_path = os.path.join(image_subdirectory, anno['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(image.size[0])
    height = int(image.size[1])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    blur = []
    expression = []
    illumination = []
    occlusion = []
    pose = []
    # TODO: invalid = []  # drop invalid image
    if 'object' in anno:
        for obj in anno['object']:
            w = float(obj['w'])
            h = float(obj['h'])
            xmin = float(obj['x1'])
            xmax = float(xmin + w)
            ymin = float(obj['y1'])
            ymax = float(ymin + h)

            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            classes_text.append('Face'.encode('utf8'))
            classes.append(label_map_dict['Face'])
            blur.append(int(obj['blur']))
            expression.append(int(obj['expression']))
            illumination.append(int(obj['illumination']))
            occlusion.append(int(obj['occlusion']))
            pose.append(int(obj['pose']))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(anno['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(anno['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/blur': dataset_util.int64_list_feature(blur),
        'image/object/expression': dataset_util.int64_list_feature(expression),
        'image/object/illumination': dataset_util.int64_list_feature(illumination),
        'image/object/occlusion': dataset_util.int64_list_feature(occlusion),
        'image/object/pose': dataset_util.int64_list_feature(pose),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dict,
                     image_dir,
                     examples):
    """
    Creation function for creating TFRecord
    :param output_filename: the name of the output
    :param num_shards: number of shards of the finish output tf record
    :param label_map_dict: label map for each image
    :param annotations_dict: annotation map for each image
    :param image_dir: image path
    :param examples: process sample data
    :return: tf.train.Example tfrecord
    """

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                 output_filename,
                                                                                 num_shards)
        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))

            # TODO: check example here
            anno = annotations_dict[str(example)]

            try:
                tf_example = dict_to_tf_example(anno,
                                                label_map_dict,
                                                image_dir)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Exception whlie writing tfrecords.')


def main(_):
    """
    Main function for generating wider face TFRecord
    """
    train_data_dir = FLAGS.train_data_dir  # train_data_dir: ./WIDER_train/images
    val_data_dir = FLAGS.val_data_dir  # val_data_dir: ./WIDER_val/images

    train_examples_path = FLAGS.train_examples_path  # train_examples_path: ./examples_files/train_examples.txt
    val_examples_path = FLAGS.val_examples_path  # val_examples_path: ./examples_files/val_examples.txt
    # TODO: check read_examples_list here
    train_examples = dataset_util.read_examples_list(train_examples_path)  # train_example: a file contains the all names of the images
    # (e.g. 0--Parade/0_Parade_marchingband_1_849.jpg)
    val_examples = dataset_util.read_examples_list(val_examples_path)  # val_example: a file contains the all names of the images
    # (e.g. 0--Parade/0_Parade_marchingband_1_465.jpg)

    train_anno_path = FLAGS.train_anno_path
    val_anno_path = FLAGS.val_anno_path
    train_anno_dict = ra.read_anno(train_anno_path)
    val_anno_dict = ra.read_anno(val_anno_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    random.seed(42)
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    # output_dir: ./tfrecord_data
    train_output_path = os.path.join(FLAGS.output_dir, 'wider_faces_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'wider_faces_val.record')

    create_tf_record(train_output_path,
                     FLAGS.num_shards,
                     label_map_dict,
                     train_anno_dict,
                     train_data_dir,
                     train_examples)

    create_tf_record(val_output_path,
                     FLAGS.num_shards,
                     label_map_dict,
                     val_anno_dict,
                     val_data_dir,
                     val_examples)


if __name__ == '__main__':
    tf.app.run()
