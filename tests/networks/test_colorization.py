import unittest

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import color

from colorization import keras_colorization, lab_to_rgb, l_to_rgb


class TestColorization(unittest.TestCase):
    def test_colorization(self):
        imgs_l, imgs_ab_true, imgs_emb = self._batch_tensors()

        # Build the network and the optimizer step
        model = keras_colorization(imgs_l, imgs_emb, imgs_ab_true)
        print(model.summary())

        self._run(model, imgs_l, imgs_emb, imgs_ab_true)

    def _run(self, model, imgs_l, imgs_emb, imgs_ab_true):
        sess = tf.Session()
        K.set_session(sess)
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            res = model.predict(
                {'imgs_l': imgs_l,
                 'imgs_emb': imgs_emb,
                 'imgs_ab_true': imgs_ab_true})

            img_gray = l_to_rgb(res[1][0][:, :, 0])
            img_output = lab_to_rgb(res[1][0][:, :, 0],
                                    res[2][0])
            img_true = lab_to_rgb(res[1][0][:, :, 0],
                                  res[3][0])

            plt.subplot(2, 3, 1)
            plt.imshow(img_gray)
            plt.title('Input (grayscale)')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.imshow(img_output)
            plt.title('Network output')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.imshow(img_true)
            plt.title('Target (original)')
            plt.axis('off')

            model.fit({'imgs_l': imgs_l,
                       'imgs_emb': imgs_emb,
                       'imgs_ab_true': imgs_ab_true},
                      {'diff': np.zeros((1, 128, 64, 2)),
                       'imgs_l': np.zeros((1, 128, 64, 1)),
                       'imgs_ab': np.zeros((1, 128, 64, 2)),
                       'imgs_ab_true': np.zeros((1, 128, 64, 2))},
                      epochs=5,
                      batch_size=16)

            res = model.predict(
                {'imgs_l': imgs_l,
                 'imgs_emb': imgs_emb,
                 'imgs_ab_true': imgs_ab_true})

            img_gray = l_to_rgb(res[1][0][:, :, 0])
            img_output = lab_to_rgb(res[1][0][:, :, 0],
                                    res[2][0])
            img_true = lab_to_rgb(res[1][0][:, :, 0],
                                  res[3][0])

            plt.subplot(2, 3, 4)
            plt.imshow(img_gray)
            plt.title('Input (grayscale)')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.imshow(img_output)
            plt.title('Network output')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.imshow(img_true)
            plt.title('Target (original)')
            plt.axis('off')

            plt.show()

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)

    def _batch_tensors(self):
        # Image sizes
        width = 128
        height = 64

        # The target image is a simple checkboard pattern
        img = np.zeros((width, height, 3), dtype=np.uint8)
        img[:width // 2, :, 0] = 255
        img[:, height // 2:, 1] = 255
        img[:width // 2, :height // 2, 2] = 255

        # Simulate a batch of Lab images with size [width, height]
        # and Lab values in the range [-1, 1]
        lab = color.rgb2lab(img).astype(np.float32)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        l = 2 * l / 100 - 1
        l = l.reshape([width, height, 1])
        ab /= 127

        imgs_l, imgs_ab_true, imgs_emb = tf.train.batch([
            tf.convert_to_tensor(l),
            tf.convert_to_tensor(ab),
            tf.truncated_normal(shape=[1001])],
            batch_size=32
        )
        return imgs_l, imgs_ab_true, imgs_emb

if __name__ == '__main__':
    unittest.main()
