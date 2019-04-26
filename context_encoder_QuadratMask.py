from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy import squeeze

import tensorflow as tf
import keras

import matplotlib.pyplot as plt

import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('Pred_img_dir', 'TESTInpaintingImagesComparision/Epoch_30_Intervall_1_batchSize_16', """Path to directory for saving pred""")
tf.app.flags.DEFINE_integer('Batch_size', 16, """batchsize of input and size of gf--the num of filters of Generator first layer """)

class ContextEncoder():
    def __init__(self):
        self.img_rows = 512
        self.img_cols = 512
        self.mask_height = 8
        self.mask_width = 8
        self.channels = 1
        self.num_classes = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def build_generator(self):

        # gf = 512 # OOM
        gf = FLAGS.Batch_size  # activation_4 (Activation)    (None, 32, 32, 1) # 128 entspricht dann batch size 128

        model = Sequential()

        # Encoder
        model.add(Conv2D(gf, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(gf * 2, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(gf * 4, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(gf * 8, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(gf * 16, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(gf * 32, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(gf*64, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # model.add(Conv2D(gf*128, kernel_size=1, strides=2, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.5))

        # Decoder

        # model.add(UpSampling2D())
        # model.add(Conv2D(gf*64, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(gf*32, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))

        # model.add(UpSampling2D())
        # model.add(Conv2D(gf*16, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))

        # TODO Unet is better to reconstruct some global features? as https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/mnist/04_CCGAN_MNIST.ipynb
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation('tanh'))

        model.summary()

        masked_img = Input(shape=self.img_shape)
        gen_missing = model(masked_img)

        return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width
        # print('x1: ', x1)
        # print('x2: ', x2)
        # print('y1: ', y1)
        # print('y2: ', y2)

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels, 1))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)  #

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = cifar10.load_data()
        train_datagen = ImageDataGenerator()
        dir_imgPredBel = '/home/yeyang/hartenbach/runs/RESNET_DILATED/watch/11_23_1103_loss=L1_f=16_s=2_d=3_ck=3_norm=ln_lr=5e-4'
        data_generator = train_datagen.flow_from_directory(dir_imgPredBel + '/ALL_PRED/', target_size=(512, 1024),
                                                           color_mode='grayscale', classes=None,
                                                           class_mode=None, batch_size=batch_size, shuffle=True,
                                                           seed=None, save_to_dir=None,
                                                           save_prefix='', save_format='png', follow_links=False,
                                                           subset=None,
                                                           interpolation='nearest')  # ich habe keine ahnung von y data hier. allerdings kann nicht als none gesetzt

        # print('data_generator shape:', data_generator.shape) # AttributeError: 'DirectoryIterator' object has no attribute 'shape'

        for epoch in range(epochs):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # print('idx is :', idx)

            # print('i in range ', (len(X_train) // batch_size))  # =1

            d_loss_epoch = []
            g_loss_epoch = []

            iterations = data_generator.n // batch_size  # IterationNum gleich BatchNum
            for i in range(iterations):
                Data_x_batch = data_generator.next()  # ZeroDivisionError: integer division or modulo by zero SOLVED BY changing the data/training to data
                # for non - classification tasks.flow_from_directory would still expect a directory that contains a subdirectory with images when class_mode is None.
                # Found 128 images belonging to 1 classes.

                # print('Data_x_batch shape:', Data_x_batch.shape)
                X_train = Data_x_batch[:, :, :512, :]  # XTrainBelO_generator
                y_train = Data_x_batch[:, :, 512:, :]  # YTrainBelF_generator
                # print('X_train shape:', X_train.shape)
                # print(X_train.shape[0], 'train samples')
                # print(X_test.shape[0], 'test samples')

                # Rescale -1 to 1
                X_train = (X_train.astype(np.float32) - 127.5) / 127.5
                X_train = np.expand_dims(X_train, axis=3)
                y_train = (y_train.astype(np.float32) - 127.5) / 127.5
                y_train = np.expand_dims(y_train, axis=3)
                # print('X_train shape:', X_train.shape)
                # print('y_train shape:', y_train.shape)

                # Adversarial ground truths
                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
                # print('valid shape:', valid.shape)

                # imgs = X_train[i * batch_size:(i + 1) * batch_size] # 0312 wrong
                imgs = X_train
                # imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], 1)) # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
                # print(' imgs shape:', imgs.shape)

                masked_imgs, missing_parts, _ = self.mask_randomly(imgs)  #
                # print(' masked_imgs shape:', masked_imgs.shape)
                # masked_imgs = masked_imgs.reshape((masked_imgs.shape[0], 64, 64, 1))
                masked_imgs.resize((masked_imgs.shape[0], self.img_rows, self.img_cols, 1))
                # print(' RESHAPED masked_imgs shape:', masked_imgs.shape)

                missing_parts = missing_parts.reshape(
                    (missing_parts.shape[0], missing_parts.shape[1], missing_parts.shape[2], 1))
                # print(' missing_parts shape:', missing_parts.shape)

                # Generate a batch of new images
                gen_missing = self.generator.predict(masked_imgs)
                # print(' gen_missing shape:', gen_missing.shape)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)  # ,
                d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])  # ,


                # # Plot the progress
                if i % (1024/batch_size) == 0:
                    print("epoch: %d batch=%d/%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
                        epoch+1 , i+1, iterations, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

            d_loss_epoch.append(d_loss)
            g_loss_epoch.append(g_loss[0])
            print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (epoch+1, epochs, d_loss[-1], g_loss[-1]), 100 * ' ')

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                samples = 3

                idx = np.random.randint(0, X_train.shape[0], samples)
                imgs_sample = X_train[idx]
                masked_imgs_sample, missing_parts_sample, (
                y1_sample, y2_sample, x1_sample, x2_sample) = self.mask_randomly(imgs_sample)
                masked_imgs_sample.resize((masked_imgs_sample.shape[0], self.img_rows, self.img_cols, 1))
                gen_missing_sample = self.generator.predict(masked_imgs_sample)
                squeezed_missing_parts_sample = missing_parts_sample.reshape(samples, 8, 8, self.channels)

                # OriginalBelF = y_train[idx]
                # OriginalBelF.resize((OriginalBelF.shape[0], self.img_rows, self.img_cols, 1))

                c = samples
                fig, axs = plt.subplots(3, c)

                for k in range(samples):

                    # plot OriginalBelF
                    # plt.subplot(c, samples, k + 1)
                    # plt.imshow(OriginalBelF[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    # plt.xticks([])
                    # plt.yticks([])

                    # plot original BelO
                    imgs_sample[k].resize(self.img_rows, self.img_cols)
                    # imgs_sample[k].squeeze(axis=2)
                    # imgs_sample[k].squeeze(axis=3)
                    # print((imgs_sample[k].reshape(self.img_rows, self.img_cols)).shape)
                    axs[0, k].imshow(imgs_sample[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    axs[0, k].axis('off')

                    # plot masked BelO
                    # masked_imgs_sample[k].resize(self.img_rows, self.img_cols)
                    # axs[1, k].imshow(masked_imgs_sample[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    # axs[1, k].axis('off')
                    # plt.subplot(c, samples, k + 1 + r)
                    # plt.imshow(masked_imgs[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    # plt.xticks([])
                    # plt.yticks([])

                    # plot missing
                    missing_imgs_sample = np.zeros((samples, self.img_rows, self.img_cols, self.channels))
                    missing_imgs_sample[k].resize(self.img_rows, self.img_cols, self.channels)

                    squeezed_missing_parts_sample[k].resize(8, 8, self.channels)

                    # print(squeezed_missing_parts_sample[k].shape)
                    # print(missing_imgs_sample.shape)
                    missing_imgs_sample[k, y1_sample[k]:y2_sample[k], x1_sample[k]:x2_sample[k], :] = squeezed_missing_parts_sample[k]
                    missing_imgs_sample[k].resize(self.img_rows, self.img_cols)

                    axs[1, k].imshow(missing_imgs_sample[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    axs[1, k].axis('off')
                    # plt.subplot(c, samples, k + 1 + 2*r)
                    # plt.imshow(missing_parts[k].reshape(8, 8), cmap='gray')
                    # plt.xticks([])
                    # plt.yticks([])

                    # plot recontructed
                    inpainted_imgs_sample = np.zeros((samples, self.img_rows, self.img_cols, self.channels))
                    inpainted_imgs_sample[k] = inpainted_imgs_sample[k].reshape(self.img_rows, self.img_cols, self.channels)
                    # print(inpainted_imgs_sample[k].shape)
                    # print(y1_sample[k], ' ', y2_sample[k], ' ', x1_sample[k], ' ', x2_sample[k], ' ')
                    # print(gen_missing_sample[k].shape)
                    inpainted_imgs_sample[k, y1_sample[k]:y2_sample[k], x1_sample[k]:x2_sample[k], :] = gen_missing_sample[k]
                    inpainted_imgs_sample[k].resize(self.img_rows, self.img_cols)

                    axs[2, k].imshow(inpainted_imgs_sample[k].reshape(self.img_rows, self.img_cols), cmap='gray')
                    axs[2, k].axis('off')
                    # plt.subplot(c, samples, k + 1 + 3*r)
                    # plt.imshow( gen_missing_sample[k].reshape(8, 8), cmap='gray')
                    # plt.xticks([])
                    # plt.yticks([])
                    #
                    #
                    #
                    # plt.savefig("InpaintingImages/%d.png" % epoch)
                    # plt.close()

                    # plt.tight_layout()
                    # plt.show()

                    # r, c = 3, 6
                    #
                    # masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
                    # print('IN SAMPLE masked_imgs shape:', masked_imgs.shape)
                    # masked_imgs.resize((masked_imgs.shape[0], 64, 64,
                    #                     1))  # I can see here that array b is not its own array, but simply a view of a (just another way to understand the "OWNDATA" flag). # ValueError: cannot reshape array of size 1572864 into shape (6,64,64,1)
                    # print('IN SAMPLE RESHAPED copied_masked_imgs shape:', masked_imgs.shape)
                    # gen_missing = self.generator.predict(masked_imgs)
                    #
                    # imgs = 0.5 * imgs + 0.5
                    # imgs.resize((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
                    # print('IN SAMPLE  imgs shape:', imgs.shape)
                    # masked_imgs = 0.5 * masked_imgs + 0.5
                    # gen_missing = 0.5 * gen_missing + 0.5
                    # print('IN SAMPLE  gen_missing shape:', gen_missing.shape)
                    #
                    # fig, axs = plt.subplots(r, c)
                    # for i in range(c):
                    #     axs[0, i].imshow(imgs[i, :, :, 0])
                    #     axs[0, i].axis('off')
                    #     axs[1, i].imshow(masked_imgs[i, :, :, 0])
                    #     axs[1, i].axis('off')
                    #     filled_in = imgs[i].copy()
                    #     filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
                    #     # axs[2,i].imshow(filled_in) # TypeError: Invalid dimensions for image data. # Here the problem was that an array of shape (nx,ny,1) is still considered a 3D array, and must be squeezed or sliced into a 2D array.
                    #
                    #     # numpy.squeeze(filled_in)
                    #     # filled_in = filled_in.array(dtype=float)
                    #     filled_in[i] = filled_in[
                    #         i].squeeze()  # without () : AttributeError: 'builtin_function_or_method' object has no attribute 'shape'
                    #     # filled_in.resize(filled_in.shape[0], filled_in.shape[1])
                    #     print('filled_in with the shape of', filled_in.shape)
                    #     filled_in = np.dtype(float)
                    #     print('filled_in with the dtype of',
                    #           type(filled_in))  # TypeError: Image data cannot be converted to float
                    #     axs[2, i].imshow(filled_in[i], cmap='gray')
                    #     axs[2, i].axis('off')
                fig.savefig("%s/%d.png" % (FLAGS.Pred_img_dir,epoch))
                plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    # # Config (for GPU usage)
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = False #TB true reche
    # sess_config.log_device_placement = False #TB not exists
    # sess = tf.Session(config=sess_config)
    # keras.backend.set_session(sess)

    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # DUMM ! NICHT HIER KONFIGUIEREN!! BUGS UNERKENNBAR!!!


    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    from keras import backend as K

    K.tensorflow_backend._get_available_gpus()

    # TODO FLAGS Epoch_30_Intervall_1
    context_encoder = ContextEncoder()

    if not os.path.exists(FLAGS.Pred_img_dir):
        os.makedirs(FLAGS.Pred_img_dir)
        # if not os.path.exists(FLAGS.logdir+"/pred"):
        #     os.makedirs(FLAGS.logdir+"/pred")
    print('Pred_img_dir is ', FLAGS.Pred_img_dir)

    context_encoder.train(epochs=5, batch_size=FLAGS.Batch_size, sample_interval=1) # 256 exceeds 10% memory of RTX2080Ti
