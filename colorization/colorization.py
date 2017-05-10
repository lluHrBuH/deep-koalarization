"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.

When using
l, emb, ab = sess.run([image_l, image_embedding, image_ab])

The function l_to_rgb converts the numpy array l into an rgb image.
The function lab_to_rgb converts the numpy arrays l and b into an rgb image.
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Input
from keras.engine import Model
from keras.layers import Conv2D, UpSampling2D, add, Lambda
from skimage import color

from fusion_layer import fusion


def keras_colorization(imgs_l, imgs_emb, imgs_ab_true):
    imgs_l = Input(tensor=imgs_l, name='imgs_l')
    imgs_emb = Input(tensor=imgs_emb, name='imgs_emb')
    imgs_ab_true = Input(tensor=imgs_ab_true, name='imgs_ab_true')

    imgs_encoded = encoder(imgs_l)
    imgs_fused = fusion(imgs_encoded, imgs_emb)
    imgs_ab = decoder(imgs_fused)

    # Hack around the loss
    neg = Lambda(lambda x: -x)(imgs_ab_true)
    diff = add([imgs_ab, neg], name='diff')

    model = Model(inputs=[imgs_l, imgs_emb, imgs_ab_true],
                  outputs=[diff, imgs_l, imgs_ab, imgs_ab_true])

    model.compile(optimizer='rmsprop',
                  loss=['mean_squared_error', ignore_loss,
                        ignore_loss, ignore_loss])
    return model


def encoder(imgs_l):
    with tf.name_scope('encoder'):
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(
            imgs_l)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    return x


def decoder(encoded):
    with tf.name_scope('decoder'):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
        imgs_ab = UpSampling2D((2, 2), name='imgs_ab')(x)
    return imgs_ab


def ignore_loss(y_true, y_pred):
    # No loss from this side
    return K.zeros_like(y_true, name='ignore_this_loss')


def l_to_rgb(img_l):
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)
