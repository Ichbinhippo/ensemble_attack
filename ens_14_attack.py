"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
# from scipy.misc import imread
# # from scipy.misc import imsave
from imageio import imread
from imageio import imsave

import scipy.stats as st
from timeit import default_timer as timer

import tensorflow as tf
from nets import resnet_v1,resnet_v2,vgg,inception_v1,inception_v2,inception_v3,inception_v4,inception_resnet_v2
import os
import pandas as pd

import PIL
import PIL.Image
from io import BytesIO
import pandas as pd

from random import randint, uniform
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2


import warnings
warnings.filterwarnings("ignore")

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')


tf.flags.DEFINE_string(
    'checkpoint_path_inception_v1', 'checkpoints/inception_v1.ckpt',
    'Path to checkpoint for resnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', 'checkpoints/inception_v3.ckpt',
    'Path to checkpoint for resnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', 'checkpoints/inception_v4.ckpt',
    'Path to checkpoint for resnet network.')

# resnet_v1

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v1_50', 'checkpoints/resnet_v1_50.ckpt',
    'Path to checkpoint for resnet network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v1_101', 'checkpoints/resnet_v1_101.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v1_152', 'checkpoints/resnet_v1_152.ckpt',
    'Path to checkpoint for inception network.')

# vgg model
tf.flags.DEFINE_string(
    'checkpoint_path_vgg_16', 'checkpoints/vgg_16.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg_19', 'checkpoints/vgg_19.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', 'checkpoints/inception_resnet_v2_2016_08_30.ckpt',
    'Path to checkpoint for inception network.')

# ens model

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_resnet_v2', 'checkpoints/adv_inception_resnet_v2.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', 'checkpoints/adv_inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', 'checkpoints/ens3_adv_inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', 'checkpoints/ens4_adv_inception_v3.ckpt',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', 'checkpoints/ens_adv_inception_resnet_v2.ckpt',
    'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'input_dir', 'images', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', 'dataset1_target_2018', 'Output directory with images.')
tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 2, 'How many images process at one time.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')
tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'sig', 2, 'gradient smoothing')
tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_integer(
    'iterations', 50, 'iterations')
tf.flags.DEFINE_float(
    'augment_stddev', 0.05, 'stddev of image_augmentation random noise.')

tf.flags.DEFINE_float(
    'rotate_stddev', 0.05, 'stddev of image_rotation random noise.')

tf.flags.DEFINE_string(
    'num_gpu', '0', 'num_gpu')
tf.flags.DEFINE_string(
    'csv_file', 'dataset2.csv', 'csv_file')

FLAGS = tf.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.num_gpu


def load_target_class(input_dir):
    """Loads target classes."""
    df = pd.read_csv(FLAGS.csv_file)
    id_names, target_class, TrueLabel = list(df['ImageId']), list(df['TargetClass']),list(df['TrueLabel'])
    return {id_names[i]: int(target_class[i])-1 for i in range(len(id_names))},{id_names[i]: int(TrueLabel[i])-1 for i in range(len(id_names))}

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
             temp = (np.asarray(PIL.Image.open(f).resize((299, 299)))).astype(np.float32)
             images[idx, :, :, :] = (2 * temp/255 ) - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
def image_augmentation(x):
    # img, noise
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=FLAGS.augment_stddev), zero, zero], axis=1)
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=FLAGS.rotate_stddev)
    return images_rotate(x, rands, interpolation='BILINEAR')

def preprocessing(x):
    x = ((x + 1.0)*0.5)*255.0
    img = tf.cast(x, tf.float32)  # 这句必须加上
    #随机设置图片的亮度
    # img = tf.image.random_brightness(img,max_delta=10)
    #随机设置图片的对比度
    # img = tf.image.random_contrast(img,lower=0.0,upper=0.5)
    #随机设置图片的饱和度
    # img = tf.image.random_saturation(img,lower=0.0,upper=0.5)
    return (img/255) * 2.0 - 1.0


def input_diversity(input_tensor):
    # input_tensor = image_augmentation(input_tensor)
    # input_tensor = preprocessing(input_tensor)
    input_tensor = image_rotation(input_tensor)
    """
    kernel_size=10
    p_dropout=0.1
    kernel = tf.divide(tf.ones((kernel_size,kernel_size,3,3),tf.float32),tf.cast(kernel_size**2,tf.float32))
    input_shape = input_tensor.get_shape()
    rand = tf.where(tf.random_uniform(input_shape) < tf.constant(p_dropout, shape=input_shape),
      tf.constant(1., shape=input_shape), tf.constant(0., shape=input_shape))
    image_d = tf.multiply(input_tensor,rand)
    image_s = tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],'SAME')
    input_tensor = tf.add(image_d,tf.multiply(image_s,tf.subtract(tf.cast(1,tf.float32),rand)))
    """
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

# preprocess
def preprocess_for_model(images, model_type):
    images = ((images + 1.0) * 0.5) * 255.0
    if 'inception_resnet_v2' in model_type.lower() or 'inception_v3' in model_type.lower() or 'inception_v4' in model_type.lower():
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'inception_v1' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    # diff: Calculate the n-th discrete difference along the given axis.
    # cdf : Cumulative distribution function evaluated at `x`
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    # outer: Compute the outer product of two vectors.
    kernel = kernel_raw / kernel_raw.sum()
    # kernel： (21,21)
    return kernel

# This is matlab version fft operation
import math
def matlab_fft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.fft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y

def matlab_ifft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.ifft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y

def dct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    y = tf.reverse(x, axis = [1])
    y = tf.concat([x, y], axis = 1)
    f_y = matlab_fft(y)
    f_y = f_y[:, 0:n, :]

    t = tf.complex(tf.constant([0.0]), tf.constant([-1.0])) * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64)
    t = tf.exp(t) / tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64)

    # since tensor obejct does not support item assignment, we have to concat a new tensor
    t0 = t[0] / tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    dct_x = W * f_y
    dct_x = tf.cast(dct_x, tf.complex64)
    dct_x = tf.real(dct_x)

    return dct_x

def idct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
    t = temp_complex * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64) * tf.exp(t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64))

    t0 = t[0] * tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    x = tf.cast(x, tf.complex64)
    yy_up = W * x
    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([-1.0]))
    yy_down = temp_complex * W[:, 1:n, :] * tf.reverse(x[:,1:n, :], axis = [1])
    yy_mid = tf.cast(tf.zeros([N, 1, m]), tf.complex64)
    yy = tf.concat([yy_up, yy_mid, yy_down], axis = 1)
    y = matlab_ifft(yy)
    y = y[:, 0:n, :]
    y = tf.real(y)
    return y

def dct2d(x):
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x

def idct2d(x):
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x

def batch_dct2d(x_in):
    x_in = (x_in + 1.0)*0.5
    # print (x_in,x_in.shape)
    x_new = tf.transpose(x_in, [0, 3, 2, 1])
    x1, x2 = tf.unstack(x_new, axis=0)

    dct2d_x1 = dct2d(x1)
    idct2d_x1 = idct2d(dct2d_x1)

    dct2d_x2 = dct2d(x2)
    idct2d_x2 = idct2d(dct2d_x2)

    dct_batch = tf.stack((dct2d_x1,dct2d_x2),axis=0)
    new_dct = tf.transpose(dct_batch, [0, 3, 2, 1])

    idct_batch = tf.stack((idct2d_x1, idct2d_x2), axis=0)
    new_idct = tf.transpose(idct_batch, [0, 3, 2, 1])
    new_idct = new_idct * 2.0 - 1.0
    # rescale_idct = tf.divide(new_idct,255.0)
    # print (new_idct)
    return new_idct

def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

from skimage.restoration import denoise_nl_means
import random
def batch_NLM(img):
    seed = random.randint(0,1)
    if seed == 0 :
        img = (img + 1.0) * 0.5
        n, w, h, c = list(img.shape)
        new_data = np.zeros_like(img)
        for i in range(n):
            new_data[i] = denoise_nl_means(img[i], multichannel=True)
        return new_data * 2. -1.0
    else:
        return img

def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    tf.logging.set_verbosity(tf.logging.INFO)

    full_start = timer()
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    all_images_taget_class,all_images_true_label = load_target_class(FLAGS.input_dir)

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        momentum = FLAGS.momentum
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = 0.2
        num_classes = 1000
        num_classes_a = 1001
        # image = x_input

        image = input_diversity(x_input)
        # image = batch_dct2d(image)

        """
        224 input
        """

        processed_imgs_res_v1_50 = preprocess_for_model(image, 'resnet_v1_50')
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
                processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        processed_imgs_res_v1_101 = preprocess_for_model(image, 'resnet_v1_101')
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_res_v1_101, end_points_res_v1_101 = resnet_v1.resnet_v1_101(
                processed_imgs_res_v1_101, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

        processed_res_v1 = preprocess_for_model(image, 'resnet_v1_152')
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits_res_v1_152, end_points_res_v1 = resnet_v1.resnet_v1_152(
                processed_res_v1, num_classes=num_classes, is_training=False, scope='resnet_v1_152',reuse=tf.AUTO_REUSE)

        processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
                processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

        processed_imgs_vgg_19 = preprocess_for_model(image, 'vgg_19')
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits_vgg_19, end_points_vgg_19 = vgg.vgg_19(
                processed_imgs_vgg_19, num_classes=num_classes, is_training=False, scope='vgg_19')

        logits_clean_a = (logits_res_v1_50 + logits_res_v1_101 + logits_res_v1_152 + logits_vgg_16 + logits_vgg_19)/5.0


        processed_imgs_inception_v1 = preprocess_for_model(image, 'inception_v1')
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            logits_inception_v1, end_points_inception_v1 = inception_v1.inception_v1(
                processed_imgs_inception_v1, num_classes=num_classes_a, is_training=False, reuse=tf.AUTO_REUSE)
        """
        299 input
        """

        x_div = preprocess_for_model(image, 'inception_v3')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
                x_div, num_classes=num_classes_a, is_training=False, scope='InceptionV3')

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
                x_div, num_classes=num_classes_a, is_training=False, scope='InceptionV4')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_div, num_classes=num_classes_a, is_training=False, scope='InceptionResnetV2')

        logits_clean_b = (logits_inc_v3 + logits_inc_v4 + logits_inc_res_v2 + logits_inception_v1)/4.0

        """
        add adv model
        """
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x_div, num_classes=num_classes_a, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_div, num_classes=num_classes_a, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_div, num_classes=num_classes_a, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_div, num_classes=num_classes_a, is_training=False, scope='EnsAdvInceptionResnetV2')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_adv_res_v2, end_points_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_div, num_classes=num_classes_a, is_training=False, scope='AdvInceptionResnetV2')

        logits_ens_a = (logits_adv_v3 + logits_ens3_adv_v3 + logits_ens4_adv_v3 + logits_ensadv_res_v2 + logits_adv_res_v2)/5.0
        logits_ens_aux = (end_points_adv_v3['AuxLogits'] + end_points_ens3_adv_v3['AuxLogits'] +
                     end_points_ens4_adv_v3['AuxLogits'] + end_points_adv_res_v2['AuxLogits'] +
                     end_points_ensadv_res_v2['AuxLogits']) / 5.0


        label_test = tf.argmax(logits_adv_v3,axis=1)
        """
        ensemble model loss
        """
        clean_logits = (logits_clean_a + logits_clean_b[:,1:1001])/2.0
        adv_logits = logits_ens_a[:,1:1001] + logits_ens_aux[:,1:1001]
        logits = (clean_logits + adv_logits)/2.0

        ens_labels = tf.argmax(logits, axis=1)

        one_hot = tf.one_hot(target_class_input, num_classes)


        loss_adv_v3 = tf.losses.softmax_cross_entropy(one_hot, logits_adv_v3[:,1:1001], label_smoothing=0.0, weights=1.0)
        loss_ens3_adv_v3 = tf.losses.softmax_cross_entropy(one_hot, logits_ens3_adv_v3[:,1:1001], label_smoothing=0.0,
                                                           weights=1.0)
        loss_ens4_adv_v3 = tf.losses.softmax_cross_entropy(one_hot, logits_ens4_adv_v3[:,1:1001], label_smoothing=0.0,
                                                           weights=1.0)
        loss_ensadv_res_v2 = tf.losses.softmax_cross_entropy(one_hot, logits_ensadv_res_v2[:,1:1001], label_smoothing=0.0,
                                                             weights=1.0)
        loss_adv_res_v2 = tf.losses.softmax_cross_entropy(one_hot, logits_adv_res_v2[:,1:1001], label_smoothing=0.0, weights=1.0)

        loss_res_v1_101 = tf.losses.softmax_cross_entropy(one_hot, logits_res_v1_101, label_smoothing=0.0, weights=1.0)
        loss_res_v1_50 = tf.losses.softmax_cross_entropy(one_hot, logits_res_v1_50, label_smoothing=0.0, weights=1.0)
        loss_vgg_16 = tf.losses.softmax_cross_entropy(one_hot, logits_vgg_16, label_smoothing=0.0, weights=1.0)
        loss_res_v1_152 = tf.losses.softmax_cross_entropy(one_hot, logits_res_v1_152, label_smoothing=0.0, weights=1.0)

        total_loss = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
        noise = tf.gradients(total_loss, x_input)[0]

        kernel = gkern(15, FLAGS.sig).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)

        noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
        # [batch, out_height, out_width, in_channels * channel_multiplier]

        noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
                                   [FLAGS.batch_size, 1, 1, 1])
        # noise = momentum * grad + noise
        noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
                                   [FLAGS.batch_size, 1, 1, 1])

        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))

        s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_101'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_152'))

        s7 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='vgg_19'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))

        s10 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionResnetV2'))
        s11 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s13 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s14 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        print('Created Graph')

        with tf.Session() as sess:
            s1.restore(sess,FLAGS.checkpoint_path_inception_v1)
            s2.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s3.restore(sess, FLAGS.checkpoint_path_inception_v4)

            s4.restore(sess, FLAGS.checkpoint_path_resnet_v1_50)
            s5.restore(sess, FLAGS.checkpoint_path_resnet_v1_101)
            s6.restore(sess, FLAGS.checkpoint_path_resnet_v1_152)

            s7.restore(sess,FLAGS.checkpoint_path_vgg_16)
            s8.restore(sess, FLAGS.checkpoint_path_vgg_19)
            s9.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)

            s10.restore(sess, FLAGS.checkpoint_path_adv_inception_resnet_v2)
            s11.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            s12.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s13.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s14.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)

            print('Initialized Models')
            processed = 0.0
            defense, tgt, untgt, final = 0.0, 0.0, 0.0, 0.0
            idx = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                target_class_for_batch = (
                        [all_images_taget_class[n[:-4]] for n in filenames]
                        + [0] * (FLAGS.batch_size - len(filenames)))
                true_label_for_batch = (
                        [all_images_true_label[n[:-4]] for n in filenames]
                        + [0] * (FLAGS.batch_size - len(filenames)))

                x_max = np.clip(images + eps, -1.0, 1.0)
                x_min = np.clip(images - eps, -1.0, 1.0)
                adv_img = np.copy(images)

                for i in range(FLAGS.iterations):
                    # loss_set = sess.run([loss_adv_v3,loss_ens3_adv_v3,loss_ens4_adv_v3,loss_ensadv_res_v2,
                    #                                               loss_adv_res_v2,loss_res_v1_101,loss_res_v1_50,loss_vgg_16,loss_res_v1_152],
                    #                                              feed_dict={x_input: batch_NLM(adv_img),
                    #                                                         target_class_input: target_class_for_batch})
                    # print ("loss:",loss_set)

                    # label_ens_model = sess.run([a,b,c,d],feed_dict={x_input: adv_img,target_class_input: target_class_for_batch})
                    # print ("label_ens_model:",label_ens_model)
                    # print (target_class_for_batch,true_label_for_batch)
                    adv_img = batch_NLM(adv_img) if i%5==0 else adv_img

                    ens_loss, pred, grad, pred_adv_v3 = sess.run([total_loss, ens_labels,noise,label_test], feed_dict={x_input: adv_img,target_class_input:target_class_for_batch})
                    adv_img = adv_img - alpha * np.clip(np.round(grad), -2, 2)
                    adv_img = np.clip(adv_img, x_min, x_max)

                    print("{} \t total_loss {}".format(i, ens_loss))
                    print ('prediction   :',pred)
                    print ('target_label :',target_class_for_batch)
                    print ('true_label   :',true_label_for_batch)

                    # print ("{} \t total_loss {} predction {} \t  target class {} \t true label  {} \t ".format(i,ens_loss,pred,target_class_for_batch,true_label_for_batch))

                    # print ("model predction {} \t  target class {} \t true label  {} \t ".format(pred,target_class_for_batch,true_label_for_batch))


                print("final prediction {} \t  target class {} \t true label  {} \t ".format(pred, target_class_for_batch,
                                                                                          true_label_for_batch))

                processed += FLAGS.batch_size
                tgt += sum(np.equal(np.array(pred), np.array(target_class_for_batch)))
                defense += sum(np.equal(np.array(pred), np.array(true_label_for_batch)))
                untgt = processed - tgt - defense
                print("processed {} \t acc {} {} \t tgt {} {} \t untgt {} {} ".format(processed, defense, defense/processed, tgt, tgt / processed, untgt,
                                                                  untgt / processed))


                full_end = timer()
                print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))


                save_images(adv_img, filenames, FLAGS.output_dir)
            print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))

if __name__ == '__main__':
    tf.app.run()
