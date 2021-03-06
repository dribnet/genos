import sys
import numpy as np
import genos.bin.train
from keras.models import model_from_json

class GenosModel:
    def __init__(self, filename=None, model=None):
        if model is not None:
            self.encoder = model.get_layer("encoder")
            self.decoder = model.get_layer("decoder")
            self.latent_dim = encoder.get_layer("input").input_shape[1]
            return

        print("loading model")
        model_file = "{}_model.json".format(filename)
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        weights_file = "{}_weights.h5".format(filename)
        self.model.load_weights(weights_file)
        print("Loaded weights from disk")

        trained_encoder = self.model.get_layer("encoder")
        trained_decoder = self.model.get_layer("decoder")
        self.latent_dim = trained_decoder.get_input_shape_at(0)[1]

        # the batch size is wired in, so let's build a new version with bs=1
        self.encoder = genos.bin.train.build_encoder_model(latent_dim=self.latent_dim, batch_size=1)
        self.encoder.set_weights(trained_encoder.get_weights())
        self.decoder = genos.bin.train.build_decoder_model(latent_dim=self.latent_dim, batch_size=1)
        self.decoder.set_weights(trained_decoder.get_weights())


    def encode_images(self, images):
        s = images.shape
        # temporary shifting from 3 channels to 1
        if s[1] == 3:
            images = images[:,1,:,:].reshape(s[0], s[2], s[3], 1)
            st_images = images
        else:
            st_images = np.stack(images, axis=3)
        # print("SHAPE: {} -> {}".format(images.shape, st_images.shape))
        [x_test_encoded, x_log_var] = self.encoder.predict(st_images, batch_size=1)
        return x_test_encoded

    def get_zdim(self):
        return self.latent_dim

    def sample_at(self, z):
        # print("SHAPE: {}".format(z.shape))
        d = self.decoder.predict(z, batch_size=1)
        s = d.shape
        if d.shape[3] == 1:
            # stack 1 channel to 3
            decoded = np.stack([d, d, d], axis=1).reshape([s[0], 3, s[1], s[2]])
        else:
            decoded = np.asarray(np.split(d,s[2],axis=2)).reshape(s[2], s[0], s[1])
        return decoded
