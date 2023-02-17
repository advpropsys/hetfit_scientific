import numpy as np
import _pickle
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input,Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

np.random.seed(4269)

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

        
class dataset:
    """ Creates dataset from input source
    """
    def __init__(self,number_samples:int, name:str):
        self.sample_size = number_samples
        self.name = name
        self.high = []
        self.low = []
        self.samples = []
        self.encoding_dim = 8
        self.latent_dim = 8
        # input data 
        self.input_data = Input(shape=(8,)) 
        
        # encoder is the encoded representation of the input
        self.encoded = Dense(self.encoding_dim*2, activation ='relu')(self.input_data)
        self.encoder_mu = keras.layers.Dense(units=self.latent_dim, name="encoder_mu")(self.encoded)
        self.encoder_log_variance = keras.layers.Dense(units=self.latent_dim, name="encoder_log_variance")(self.encoded)
        
        def sampling(mu_log_variance):
            mu, log_variance = mu_log_variance
            epsilon = keras.backend.random_normal(shape=keras.backend.shape(mu), mean=0.0, stddev=1.0)
            random_sample = mu + keras.backend.exp(log_variance/2) * epsilon
            return random_sample

        encoder_output = keras.layers.Lambda(sampling, name="encoder_output")([self.encoder_mu, self.encoder_log_variance])

        self.encoder = keras.models.Model(self.input_data, encoder_output, name="encoder_model")
        #----------------------------------------------------------------
        self.latent_inputs = keras.Input(shape=(self.latent_dim,))
        self.decoded = Dense(self.encoding_dim,activation='sigmoid')(self.latent_inputs)
        self.decoder = Model(self.latent_inputs,self.decoded,name="decoder")
        
    def loss_func(self,encoder_mu, encoder_log_variance):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = keras.backend.mean(keras.backend.square(y_true-y_predict))
            return reconstruction_loss_factor * reconstruction_loss

        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance))
            print(kl_loss)
            return kl_loss

        def vae_kl_loss_metric(y_true, y_predict):
            kl_loss = -0.5 * keras.backend.sum(1.0 + encoder_log_variance - keras.backend.square(encoder_mu) - keras.backend.exp(encoder_log_variance))
            return kl_loss

        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)

            loss = reconstruction_loss + kl_loss
            return loss
        return vae_loss
    
    def generate(self):
        self.vae_input = Input(shape=(8,), name="VAE_input")
        self.vae_encoder_output = self.encoder(self.vae_input)
        self.vae_decoder_output = self.decoder(self.vae_encoder_output)
        self.vae = Model(self.vae_input, self.vae_decoder_output)
        self.vae.compile(optimizer='Adam',loss=self.loss_func(self.encoder_mu, self.encoder_log_variance))
        # self.vae.fit(mnist_digits, epochs=30, batch_size=128)
        # self.vae.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        with open(r"./data/dataset.csv", "rb") as input_file:
            local = pd.read_csv(input_file)
            local = local.iloc[:,1:]
            x_train, x_test, = train_test_split(local, test_size=0.1, random_state=42)
            
        for name in local.columns:
            self.high.append(max(local[f'{name}']))
            self.low.append(min(local[f'{name}']))
            
        self.vae.fit(x_train,
                x_train,
                epochs=50,
                batch_size=2,
                shuffle=True,
                use_multiprocessing=True)

        
        encoded_data = self.encoder.predict(x_test)
        decoded_data = self.decoder.predict(encoded_data)
        
        print(self.vae.summary(),decoded_data)
        # samples=pd.DataFrame(self.samples).T
        # local = pd.concat([local,samples],ignore_index=True)
        return None
                
            