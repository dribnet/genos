import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D
from keras.models import Model, load_model, model_from_json
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
import tensorflow as tf

def sampling(args, batch_size, latent_dim, epsilon_std):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def build_vae_loss(img_rows, img_cols, z_log_var, z_mean):
    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    return vae_loss

def build_model(batch_size, encoder_only=False):
    # input image dimensions
    img_rows, img_cols, img_chns = 28, 28, 1
    # number of convolutional filters to use
    nb_filters = 64
    # convolution kernel size
    nb_conv = 3

    original_img_size = (img_rows, img_cols, img_chns)
    latent_dim = 2
    intermediate_dim = 128
    epsilon_std = 0.01

    x = Input(batch_shape=(batch_size,) + original_img_size)
    conv_1 = Convolution2D(img_chns, 2, 2, border_mode='same', activation='relu', dim_ordering='tf')(x)
    conv_2 = Convolution2D(nb_filters, 2, 2,
                           border_mode='same', activation='relu', dim_ordering='tf',
                           subsample=(2, 2))(conv_1)
    conv_3 = Convolution2D(nb_filters, nb_conv, nb_conv,
                           border_mode='same', activation='relu', dim_ordering='tf',
                           subsample=(1, 1))(conv_2)
    conv_4 = Convolution2D(nb_filters, nb_conv, nb_conv,
                           border_mode='same', activation='relu', dim_ordering='tf',
                           subsample=(1, 1))(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim, name="z_mean")(hidden)
    z_log_var = Dense(latent_dim, name="z_log_var")(hidden)

    if encoder_only:
        return Model(x, z_mean)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,), arguments= {
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "epsilon_std": epsilon_std
        })([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(nb_filters * 14 * 14, activation='relu')

    output_shape = (batch_size, 14, 14, nb_filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                       output_shape,
                                       border_mode='same',
                                       subsample=(1, 1),
                                       activation='relu', dim_ordering='tf')
    decoder_deconv_2 = Deconvolution2D(nb_filters, nb_conv, nb_conv,
                                       output_shape,
                                       border_mode='same',
                                       subsample=(1, 1),
                                       activation='relu', dim_ordering='tf')
    output_shape = (batch_size, 29, 29, nb_filters)
    decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, 2, 2,
                                              output_shape,
                                              border_mode='valid',
                                              subsample=(2, 2),
                                              activation='relu', dim_ordering='tf')
    decoder_mean_squash = Convolution2D(img_chns, 2, 2,
                                        border_mode='valid',
                                        activation='sigmoid', dim_ordering='tf')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    vae = Model(x, x_decoded_mean_squash)
    return vae

def plot_history(subdir, history):
    outplot = "{}/history.png".format(subdir)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outplot)

def load_dataset(fname):
    X = HDF5Matrix(fname, 'X')
    y = HDF5Matrix(fname, 'y')
    return [X, y]

def main():
    parser = argparse.ArgumentParser()
    #general model params
    parser.add_argument('--batch-size', dest='batch_size', default=200, type=int,
                        help='batch-size')
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='epochs')
    parser.add_argument('--save', dest='save', default=None,
                        help='model+weights to save')
    parser.add_argument('--model', dest='model', default=None,
                        help='model file to save')
    parser.add_argument('--weights', dest='weights', default=None,
                        help='weights file to save')
    parser.add_argument('--load', dest='load', default=None,
                        help='model+weights to load')
    parser.add_argument('--oldmodel', dest='oldmodel', default=None,
                        help='start from old model')
    parser.add_argument('--oldweights', dest='oldweights', default=None,
                        help='start from old weights')
    parser.add_argument('--save-every-epoch', dest="save_every_epoch", default=False, action='store_true',
                        help="Save model every epoch")
    parser.add_argument('--datafile', dest='datafile', default=None,
                        help='used single datafile for train/test data')
    parser.add_argument('--input-glob', dest='input_glob', default=None,
                        help='glob for train/test data')
    parser.add_argument('--train-glob', dest='train_glob', default=None,
                        help='glob for train data')
    parser.add_argument('--test-glob', dest='test_glob', default=None,
                        help='glob for test data')
    parser.add_argument('--train-dataset', dest='train_dataset', default=None,
                        help='baked train-dataset file')
    parser.add_argument('--test-dataset', dest='test_dataset', default=None,
                        help='baked test-dataset file')
    parser.add_argument('--bake-dataset', dest='bake_dataset', default=None,
                        help='location to save dataset as baked npz files')
    parser.add_argument("--reconstruction-factor", type=float,
                        dest="reconstruction_factor", default=1.0,
                        help="Scaling Factor for reconstruction term")
    parser.add_argument("--discriminative-factor", type=float,
                        dest="discriminative_factor", default=1.0,
                        help="Scaling Factor for discriminative term")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Dataset for training.")
    parser.add_argument("--subdir", dest='subdir', type=str, default="outputs",
                        help="Subdirectory for output files (images)")
    parser.add_argument("--kl-factor", type=float, dest="kl_factor",
                        default=1.0, help="Scaling Factor for KL term")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--graph-latent', dest="graph_latent", default=False, action='store_true',
                        help="Graph latent space")
    parser.add_argument('--graph-grid', dest="graph_grid", default=False, action='store_true',
                        help="Grid of latent space")
    parser.add_argument('--save-weights', dest="save_weights", default=False, action='store_true',
                        help="save weights to files")
    args = parser.parse_args()

    if args.input_glob is not None:
        if args.train_glob is None:
            args.train_glob = args.input_glob
        if args.test_glob is None:
            args.test_glob = args.input_glob

    # set to None means "nothing to train"
    x_train = None
    if args.train_glob is not None:
        print("loading training files")
        x_train, y_train = dataloader.load_glob(args.train_glob)
        if args.bake_dataset is not None:
            fname = "{}_train.h5".format(args.bake_dataset)
            print("saving {}".format(fname))
            save_dataset(fname, X=x_train, y=y_train)

    if args.test_glob is not None:
        print("loading test files")
        x_test, y_test = dataloader.load_glob(args.test_glob)
        if args.bake_dataset is not None:
            fname = "{}_test.h5".format(args.bake_dataset)
            print("saving {}".format(fname))
            save_dataset(fname, X=x_test, y=y_test)

    if args.train_dataset is not None:
        print("loading train data from dataset")
        x_train, y_train = load_dataset(args.train_dataset)
        print("train: loaded {} rows".format(len(x_train)))

    if args.test_dataset is not None:
        print("loading test data from dataset")
        x_test, y_test = load_dataset(args.train_dataset)
        print("test: loaded {} rows".format(len(x_test)))

    ### SETUP MODEL
    oldmodel = args.oldmodel
    oldweights = args.oldweights
    if args.load is not None:
        oldmodel = "{}_model.json".format(args.load)
        oldweights = "{}_weights.h5".format(args.load)

    save_model_filename = args.model
    save_weights_filename = args.weights
    if args.save is not None:
        save_model_filename = "{}_model.json".format(args.save)
        save_weights_filename = "{}_weights.h5".format(args.save)

    if oldmodel is not None:
        print("loading model")
        json_file = open(oldmodel, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        print("building model")
        model = build_model(args.batch_size)

    # this is how you wire the model up and compile
    img_rows, img_cols, img_chns = 28, 28, 1
    z_log_var = model.get_layer("z_log_var").output
    z_mean = model.get_layer("z_mean").output
    model_loss = build_vae_loss(img_rows, img_cols, z_log_var, z_mean)
    model.compile(optimizer='adam', loss=model_loss)

    if save_model_filename is not None:
        # if we want to save the model, the go ahead and do this
        model_json = model.to_json()
        with open(save_model_filename, "w") as json_file:
            json_file.write(model_json)

    if oldweights is not None:
        model.load_weights(oldweights)
        print("Loaded weights from disk")

    # train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()

    img_rows, img_cols, img_chns = 28, 28, 1
    original_img_size = (img_rows, img_cols, img_chns)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    print('x_train.shape:', x_train.shape, 'x_test.shape:', x_test.shape)

    callbacks = []
    if save_weights_filename is not None:
        callbacks = [
            ModelCheckpoint(filepath=save_weights_filename, save_weights_only=True, verbose=1, save_best_only=True),
        ]
    if args.save_every_epoch:
        callbacks.append(ModelCheckpoint(filepath="{}_cur".format(save_weights_filename), verbose=1, save_best_only=False))

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    callbacks.append(early_stopping)

    # print("{} and {}".format(batch_size, args.batch_size))
    if x_train is not None and args.epochs > 0:
        print("training model")
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_test, x_test),
            batch_size=args.batch_size,
            shuffle='batch',
            nb_epoch=args.epochs,
            # verbose=2,
            callbacks=callbacks)
        plot_history(args.subdir, history)

    # model.fit(x_train, x_train,
    #     validation_data=(x_test, x_test),
    #     shuffle=True, nb_epoch=100, batch_size=batch_size, verbose=2,
    #     callbacks=[early_stopping])

    if args.graph_latent:
        # build a model to project inputs on the latent space
        encoder = build_model(args.batch_size, encoder_only=True)

        # display a 2D plot of the digit classes in the latent space
        x_test_encoded = encoder.predict(x_test, batch_size=args.batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        plt.savefig("{}/latent.png".format(args.subdir), bbox_inches='tight')

    if args.graph_grid:
        batch_size = 1

        _hid_decoded = Dense(intermediate_dim, activation='relu')
        _up_decoded = Dense(nb_filters * 14 * 14, activation='relu')
        _reshape_decoded = Reshape((14, 14, nb_filters))
        _deconv_1_decoded = Deconvolution2D(nb_filters, nb_conv, nb_conv, (batch_size, 14, 14, nb_filters),
                                            border_mode='same', subsample=(1, 1), activation='relu', dim_ordering='tf')
        _deconv_2_decoded = Deconvolution2D(nb_filters, nb_conv, nb_conv, (batch_size, 14, 14, nb_filters),
                                            border_mode='same', subsample=(1, 1), activation='relu', dim_ordering='tf')
        _x_decoded_relu = Deconvolution2D(nb_filters, 2, 2, (batch_size, 29, 29, nb_filters),
                                          border_mode='valid', subsample=(2, 2), activation='relu', dim_ordering='tf')
        _x_decoded_mean_squash = Convolution2D(img_chns, 2, 2, border_mode='valid', activation='sigmoid', dim_ordering='tf')

        decoder_input = Input(shape=(latent_dim,))
        layer1 = _hid_decoded(decoder_input)
        layer2 = _up_decoded(layer1)
        layer3 = _reshape_decoded(layer2)
        layer4 = _deconv_1_decoded(layer3)
        layer5 = _deconv_2_decoded(layer4)
        layer6 = _x_decoded_relu(layer5)
        layer7 = _x_decoded_mean_squash(layer6)
        generator = Model(decoder_input, layer7)

        _hid_decoded.set_weights(decoder_hid.get_weights())
        _up_decoded.set_weights(decoder_upsample.get_weights())
        _deconv_1_decoded.set_weights(decoder_deconv_1.get_weights())
        _deconv_2_decoded.set_weights(decoder_deconv_2.get_weights())
        _x_decoded_relu.set_weights(decoder_deconv_3_upsamp.get_weights())
        _x_decoded_mean_squash.set_weights(decoder_mean_squash.get_weights())

        # display a 2D manifold of the digits
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        from scipy.stats import norm

        # we will sample n points within [-1.5, 1.5] standard deviations
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
                x_decoded = generator.predict(z_sample, batch_size=batch_size)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig("{}/grid.png".format(args.subdir), bbox_inches='tight')

    if args.save_weights:
        WEIGHTS_FILEPATH = 'mnist_vae.hdf5'
        MODEL_ARCH_FILEPATH = 'mnist_vae.json'

        generator.save_weights(WEIGHTS_FILEPATH)

        with open(MODEL_ARCH_FILEPATH, 'w') as f:
            f.write(generator.to_json())

if __name__ == '__main__':
    main()
