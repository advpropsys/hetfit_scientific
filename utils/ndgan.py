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
        self.latent=16
        self.dfs=data
        print(data.head())
        self.inputs=8
        self.outputs=8
        self.generator_model, self.discriminator_model = self.build_models()
        self.gan_model = self.define_gan(self.generator_model,self.discriminator_model)
        
        
    def define_discriminator(self,inputs=8):
        ''' function to return the compiled discriminator model'''
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
        model = Sequential()
        model.add(Dense(20, activation = 'relu', kernel_initializer= 'he_uniform', input_dim = latent_dim))
        model.add(LeakyReLU(alpha = 0.3))
        model.add(Dense(15, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(self.outputs, activation = 'elu'))
        return model
    
    def build_models(self):
        discriminator_model = self.define_discriminator()
        generator_model = self.define_generator(self.latent)
        return generator_model,discriminator_model
    
    def generate_latent_points(self,latent_dim, n):
        '''generate points in latent space as input for the generator'''
        x_input = np.random.rand(latent_dim*n) #generate points in latent space
        x_input = x_input.reshape(n,latent_dim)  #reshape
        return x_input

    def generate_fake_samples(self,generator, latent_dim, n):
        x_input = self.generate_latent_points(latent_dim, n) #genarate points in latent space
        x = generator.predict(x_input) #predict outputs
        y = np.zeros((n, 1))
        return x, y
    
    def define_gan(self,generator, discriminator):
        '''define the combined generator and discriminator model'''
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        return model
    
    
    def summarize_performance(self,epoch, generator, discriminator, latent_dim, n = 200):
        '''evaluate the discriminator and plot real and fake samples'''
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
        ''' function to train gan model'''
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
        self.train_gan(self.generator_model, self.discriminator_model, self.gan_model, self.latent)
        
    def predict(self,n):
        x_fake, y_fake = self.generate_fake_samples(self.generator_model, self.latent, n)
        return x_fake