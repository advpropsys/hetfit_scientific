from keras.models import Sequential
from keras.layers import Dense, LeakyReLU,ReLU
# from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4062710504)
from keras.utils import set_random_seed
set_random_seed(440)

class DCGAN():
    def __init__(self,latent,data):
        """
        The function takes in two arguments, the latent space dimension and the dataframe. It then sets
        the latent space dimension, the dataframe, the number of inputs and outputs, and then builds the
        models
        
        :param latent: The number of dimensions in the latent space
        :param data: This is the dataframe that contains the data that we want to generate
        """
        self.latent=16
        self.dfs=data
        print(data.head())
        self.inputs=8
        self.outputs=8
        self.generator_model, self.discriminator_model = self.build_models()
        self.gan_model = self.define_gan(self.generator_model,self.discriminator_model)
        
        
    def define_discriminator(self,inputs=8):
        """
        The discriminator is a neural network that takes in a vector of length 8 and outputs a single
        value between 0 and 1
        
        :param inputs: number of features in the dataset, defaults to 8 (optional)
        :return: The model is being returned.
        """
        model = Sequential()
        model.add(Dense(20, activation = 'relu', kernel_initializer = 'he_uniform', input_dim = self.inputs))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(15, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(5, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation = 'selu'))
        model.compile(optimizer = 'adam', loss = 'binary_focal_crossentropy', metrics = ['accuracy'])
        return model

    def define_generator(self,latent_dim, outputs = 8):
        """
        The function takes in a latent dimension and outputs and returns a model with two hidden layers
        and an output layer
        
        :param latent_dim: The dimension of the latent space, or the space that the generator will map
        to
        :param outputs: the number of outputs of the generator, defaults to 8 (optional)
        :return: The model is being returned.
        """
        model = Sequential()
        model.add(Dense(20, activation = 'relu', kernel_initializer= 'he_uniform', input_dim = latent_dim))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Dense(15, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.outputs, activation = 'elu'))
        return model
    
    def build_models(self):
        """
        The function returns the generator and discriminator models
        :return: The generator and discriminator models are being returned.
        """
        discriminator_model = self.define_discriminator()
        generator_model = self.define_generator(self.latent)
        return generator_model,discriminator_model
    
    def generate_latent_points(self,latent_dim, n):
        """
        > Generate random points in latent space as input for the generator
        
        :param latent_dim: the dimension of the latent space, which is the input to the generator
        :param n: number of images to generate
        :return: A numpy array of random numbers.
        """
        x_input = np.random.rand(latent_dim*n) #generate points in latent space
        x_input = x_input.reshape(n,latent_dim)  #reshape
        return x_input

    def generate_fake_samples(self,generator, latent_dim, n):
        """
        It generates a batch of fake samples with class labels
        
        :param generator: The generator model that we will train
        :param latent_dim: The dimension of the latent space, e.g. 100
        :param n: The number of samples to generate
        :return: x is the generated images and y is the labels for the generated images.
        """
        x_input = self.generate_latent_points(latent_dim, n) #genarate points in latent space
        x = generator.predict(x_input) #predict outputs
        y = np.zeros((n, 1))
        return x, y
    
    def define_gan(self,generator, discriminator):
        """
        The function takes in a generator and a discriminator, sets the discriminator to be untrainable,
        and then adds the generator and discriminator to a sequential model. The sequential model is then compiled with an optimizer and a loss function. 
        
        The optimizer is adam, which is a type of gradient descent algorithm. 
        
        Loss function is binary crossentropy, which is a loss function that is used for binary
        classification problems. 

        
        The function then returns the GAN. 
        :param generator: The generator model
        :param discriminator: The discriminator model that takes in a dataset and outputs a single value
        representing fake/real
        :return: The model is being returned.
        """
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        return model
    
    
    def summarize_performance(self,epoch, generator, discriminator, latent_dim, n = 200):
        """
        > This function evaluates the discriminator on real and fake data, and plots the real and fake
        data
        
        :param epoch: the number of epochs to train for
        :param generator: the generator model
        :param discriminator: the discriminator model
        :param latent_dim: The dimension of the latent space
        :param n: number of samples to generate, defaults to 200 (optional)
        """
        x_real, y_real = self.dfs.iloc[:,1:].values, np.ones((23, 1))
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose = 1)
        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, n)
        _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose = 1)
        print('Epoch: ' + str(epoch) + ' Real Acc.: ' + str(acc_real) + ' Fake Acc.: '+ str(acc_fake))
        # x_real /= np.max(np.abs(x_real),axis=0)
        plt.scatter(x_real[:,0], x_real[:,1], color = 'red')
        plt.scatter(x_fake[:,0], x_fake[:,1], color = 'blue',s=20)
        plt.show()
        
    def train_gan(self,g_model,d_model,gan_model,latent_dim, num_epochs = 2500,num_eval = 2500, batch_size = 2):
        """
        
        :param g_model: the generator model
        :param d_model: The discriminator model
        :param gan_model: The GAN model, which is the generator model combined with the discriminator
        model
        :param latent_dim: The dimension of the latent space. This is the number of random numbers that
        the generator model will take as input
        :param num_epochs: The number of epochs to train for, defaults to 2500 (optional)
        :param num_eval: number of epochs to run before evaluating the model, defaults to 2500
        (optional)
        :param batch_size: The number of samples to use for each gradient update, defaults to 2
        (optional)
        """
        
        half_batch = 1
        #run epochs 
        for i in range(num_epochs):
            X_real, y_real = self.dfs.iloc[:,1:].values, np.ones((23, 1)) #generate real examples
            d_model.train_on_batch(X_real, y_real)               # train on real data
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch) #generate fake samples
            d_model.train_on_batch(X_fake, y_fake)                #train on fake data
            #prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points(latent_dim, batch_size)
            y_gan = np.ones((batch_size, 1))    #generate fake labels for gan
            gan_model.train_on_batch(x_gan, y_gan)
            if (i+1) % num_eval == 0:
                self.summarize_performance(i + 1, g_model, d_model, latent_dim)
                
    def start_training(self):
        """
        The function takes the generator, discriminator, and gan models, and the latent vector as
        arguments, and then calls the train_gan function.
        """
        self.train_gan(self.generator_model, self.discriminator_model, self.gan_model, self.latent)
        
        
    def predict(self,n):
        """
        It takes the generator model and the latent space as input and returns a batch of fake samples
        
        :param n: the number of samples to generate
        :return: the generated fake samples.
        """
        x_fake, y_fake = self.generate_fake_samples(self.generator_model, self.latent, n)
        return x_fake