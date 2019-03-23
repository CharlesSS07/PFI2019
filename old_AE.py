import tensorflow as tf
import numpy as np

from datetime import datetime
import subprocess
import os
#import copy
#import proteinLoader # a package I made for loading the protein data
#import PathManager#.PathManager

#os.environ['CUDA_VISIBLE_DEVICES'] = '' # uncomment for cpu, no gpu
#print('set to use cpu')



class Layer():
    
    def __init__(self, shape, genfunc, name):
        '''
        class for dealing with layer stuff, including type, size and inputs
        '''
        self.shape = shape
        self.genfunc = genfunc
        self.name = name
        
class LayerManager():
    
    def __init__(self, layers, begin_shape, end_shape):
        
        self.layers = layers
        self.bshape = begin_shape
        self.eshape = end_shape
        self.model = {}
        self.previous = ''
        
    def generate_layers(self, X):
        
        self.model[X.name] = X
        l = self.layers.pop(0)
        self.model[l.name] = l.genfunc(self.model[X.name], self.bshape, l.shape, l.name)
        self.previous = l
        self.generate_layer()
        
    def generate_layer(self):
        if len(self.layers)>=2:
            l = self.layers.pop(0)
            p = self.previous
            self.model[l.name] = l.genfunc(self.model[p.name], p.shape, l.shape, l.name)
            self.previous = l
            self.generate_layer()
        elif len(self.layers)==1:
            l = self.layers.pop(0)
            p = self.previous
            self.model[l.name] = l.genfunc(self.model[p.name], p.shape, l.shape, l.name)
            self.previous = l

class Autoencoder():
    
    def __init__(self, paths, datas, learn_rate=0.00001, learn_rate_fac=0.0001):
        print('this is the current version')
        self.paths = paths#PathManager(name=name, is_recording=recording, data_folder=data_folder)
        self.paths.save_file(__file__)
        
        self.np_seed = 1#4155576889 # setting all seeds for testing purposes, makes results reproducible.
        print('seed:', self.np_seed)
        self.tf_seed = self.np_seed        # seed for all operations
        tf.set_random_seed(self.tf_seed) # sets tf graph level seed # also set after sess is created
        np.random.seed(self.np_seed)
        
        self.datas = datas # DataManager(paths, batch_size=batch_size)
        
        #self.units = [4000, 2000, 4000]# size of each layer of autoencoder (not counting nonhidden layers)
        
        # edited to suit human data better
        
        gpu_options = tf.GPUOptions(allow_growth=True) # allows for dynamic memory allocation on GPU # tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))#tf.Session() # instanciates a tf Session with special gpu options to do dynamic memory, instead of using all avaliable memory
        
        self.learn_rate = learn_rate
        
        self.X = tf.placeholder(
            dtype=tf.float32, 
            shape=(None, self.datas.interaction_data.shape[1]), 
            name='x'
        )
        
        self.training = tf.placeholder(
            dtype=tf.bool, 
            shape=(), 
            name='training'
        )
        
        self.feed_testing = {
            self.training: False,
            self.X: self.datas.testing_data#interaction_data
        }
        
        self.feed_training = {
            self.training: False,
            self.X: self.datas.training.next_batch()#interaction_data
        }
        
        self.feed_dropout = {
            self.training: True,
            self.X: self.datas.training.next_batch()#interaction_data
        }
        
        layers = [
            #Layer(self.datas.interaction_data.shape[1], self.relu_dense_layer,    'input'),
            Layer(4000,                                 self.relu_dense_layer,    'hidden0'),
            Layer(2000,                                 self.relu_dense_layer,    'latent'),
            Layer(4000,                                 self.relu_dense_layer,    'hidden1'),
            Layer(self.datas.interaction_data.shape[1], self.relu_dense_layer,    'output')
        ]
        
        #self.random_uniform_initializer_fac = tf.initializers.random_uniform(minval=0, maxval=1, seed=self.tf_seed)
        #self.random_uniform_initializer = tf.initializers.random_uniform(minval=-1, maxval=1, seed=self.tf_seed)
        self.rescaled = self.rescale(self.X)
        self.lm = LayerManager(layers, self.datas.interaction_data.shape[1], self.datas.interaction_data.shape[1]) # inputs same as outputs because autoencoder
        self.lm.generate_layers(self.rescaled)
        self.model = {}
        self.model['hiddel_0'] = self.lm.model['latent']
        #self.fac_vars = [] # isolated for training seperately
        
        #print(self.lm.model.keys())
        self.loss_crossentropy = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.nn.sigmoid(self.rescaled),  # use sigmoid to normalize correctly
                logits=tf.nn.sigmoid(self.lm.model['output'])
            )
        ) # find loss between rescaled and output
        self.loss_meansquared = tf.losses.mean_squared_error(
            self.rescaled, 
            self.lm.model['output']
        )
        
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_step_mse = self.optimizer.minimize(loss=self.loss_meansquared) # optimize mse when run
        self.train_step_sce = self.optimizer.minimize(loss=self.loss_crossentropy) # optimize sigmoid cross entropy loss when run
        '''self.train_step_sce_mix_fac = tf.train.AdamOptimizer(learn_rate_fac).minimize(
            loss=self.loss_crossentropy, 
            var_list=self.fac_vars
        ) # optimize sigmoid cross entropy loss when run on only the mix layer'''
        
        if self.paths.is_recording:
            self.summary_writer_training = tf.summary.FileWriter(
                os.path.join(self.paths.model_folder, 'training'), 
                self.sess.graph
            )
            self.summary_writer_testing = tf.summary.FileWriter(
                os.path.join(self.paths.model_folder, 'testing'), 
                self.sess.graph
            )
        else: # put into uselses place, because summary writers will still be used
            self.summary_writer_training = tf.summary.FileWriter(
                os.path.join('/tmp/', self.paths.model_folder, 'training'), 
                self.sess.graph
            )
            self.summary_writer_testing = tf.summary.FileWriter(
                os.path.join('/tmp/', self.paths.model_folder, 'testing'), 
                self.sess.graph
            )
        
        #with tf.variable_scope('input', reuse=True):
        #    mix_fac = tf.get_variable(name='mix_fac')
        
        self.tf_summary = tf.summary.merge([ # what to write out to tensorboard for plotting
            tf.summary.scalar('CrossEntropy', self.loss_crossentropy),
            tf.summary.scalar('MeanSquared', self.loss_meansquared)#,
            #tf.summary.histogram('mix_fac_input', mix_fac)
        ])
        
        #run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        #md = tf.RunMetadata()
        #self.summary_writer_testing.add_run_metadata(md, 'test', 0)
        self.sess.run(tf.global_variables_initializer())#, options=run_options, run_metadata=md)
        #self.summary_writer_testing.add_run_metadata(md, 'test', 1)
        self.saver = tf.train.Saver()
        
        self.paths.is_old(self.restore) # runs self.restore() only if is this is a trained model
        
        print('success')
    
    def restore(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.paths.model_path)

    def rescale(self, inputs, name='rescale'):# rescales data according to medain, min and std
        '''
        subtract median of ENTIRE dataset
        divide by standard deviation of ENTIRE dataset
        subtract by minimum of ENTIRE dataset
        '''
        
        tf_MEDIAN = tf.constant(self.datas.median,  dtype=tf.float32, name="MEDIAN")
        tf_STD = tf.constant(self.datas.std,     dtype=tf.float32, name="STD")
        tf_MIN = tf.constant(self.datas.minimum, dtype=tf.float32, name="MIN")
        
        with tf.variable_scope(name):
            return tf.subtract(tf.divide(tf.subtract(inputs, tf_MEDIAN), tf_STD),  tf_MIN)
    
    '''
    def dense_layer(self, inputs, start_units, end_units, name='sigmoid_relu_hidden'):
        
        #neater way of initializing layers, automatically sets up vars/names, and more readable
        #also creates a droprate variable for the layer, under 'LAYERNAME/droprate'
        
        with tf.variable_scope(name, dtype=tf.float32):
            random_uniform_initializer_fac = tf.initializers.random_uniform(minval=0, maxval=1, seed=self.tf_seed)
            random_uniform_initializer = tf.initializers.random_uniform(minval=-1, maxval=1, seed=self.tf_seed)
            
            tf.get_variable( # sets up tf vars correctly
                name='dense_sigmoid/kernel', 
                shape=(start_units, end_units), 
                initializer=random_uniform_initializer
            )
            tf.get_variable(
                name='dense_sigmoid/bias', 
                shape=(end_units), 
                initializer=tf.zeros_initializer
            )
            
            tf.get_variable(
                name='dense_relu/kernel', 
                shape=(start_units, end_units), 
                initializer=random_uniform_initializer
            )
            tf.get_variable(
                name='dense_relu/bias', 
                shape=(end_units), 
                initializer=tf.zeros_initializer
            )
            
            fac = tf.get_variable(
                name='mix_fac', 
                shape=(end_units), 
                initializer=random_uniform_initializer_fac, 
                constraint=lambda x: tf.clip_by_value(x, 0, 1)
            )#tf.constant_initializer(0.5)
            #self.fac_vars.append(fac)
            
            droprate = tf.placeholder(dtype=tf.float32, shape=(), name='droprate')
            sigmoid_layer = tf.layers.dense(inputs=inputs, units=end_units, activation=tf.nn.sigmoid, name='dense_sigmoid', reuse=True)
            relu_layer = tf.layers.dense(inputs=inputs, units=end_units, activation=tf.nn.relu, name='dense_relu', reuse=True)
            
            #inputs = tf.contrib.layers.batch_norm(inputs=inputs, is_training=self.training, scale=True)
            inputs = fac*(sigmoid_layer-relu_layer)+relu_layer
            
            self.feed_testing[droprate] = 1
            self.feed_dropout[droprate] = 0.5
            #self.feed_full[droprate] = 1 is not used any where else
            self.feed_training[droprate] = 1

            return tf.layers.dropout(inputs=inputs, rate=droprate, training=self.training, seed=self.tf_seed)
    '''
    
    def sigmoid_dense_layer(self, inputs, start_units, end_units, name='sigmoid_hidden'):
        '''
        neater way of initializing layers, automatically sets up vars/names, and more readable
        also creates a droprate variable for the layer, under 'LAYERNAME/droprate'
        '''
        with tf.variable_scope(name, dtype=tf.float32):
            
            tf.get_variable( # sets up tf vars correctly
                name='dense_sigmoid/kernel', 
                shape=(start_units, end_units), 
                initializer=tf.zeros_initializer,
                dtype=tf.float32
            )
            tf.get_variable(
                name='dense_sigmoid/bias', 
                shape=(end_units), 
                initializer=tf.zeros_initializer,
                dtype=tf.float32
            )
            
            sigmoid_layer = tf.layers.dense(
                inputs=inputs, 
                units=end_units, 
                activation=tf.nn.sigmoid, 
                name='dense_sigmoid', 
                reuse=True
            )
            
            droprate = tf.placeholder(dtype=tf.float32, shape=(), name='droprate')
            #batch_norm = tf.contrib.layers.batch_norm(inputs=inputs, is_training=self.training, scale=True)
            
            self.feed_testing[droprate] = 1 # setting droprate for each layer
            self.feed_dropout[droprate] = 0.5
            #self.feed_full[droprate] = 1 # is not used any more
            self.feed_training[droprate] = 1
            
            return tf.layers.dropout(inputs=sigmoid_layer, rate=droprate, training=self.training, seed=self.tf_seed)
    
    def relu_dense_layer(self, inputs, start_units, end_units, name='relu_hidden'):
        '''
        neater way of initializing layers, automatically sets up vars/names, and more readable
        also creates a droprate variable for the layer, under 'LAYERNAME/droprate'
        '''
        with tf.variable_scope(name, dtype=tf.float32):
            
            tf.get_variable( # sets up tf vars correctly
                name='dense_relu/kernel', 
                shape=(start_units, end_units), 
                initializer=tf.zeros_initializer, 
                dtype=tf.float32
            )
            tf.get_variable(
                name='dense_relu/bias', 
                shape=(end_units), 
                initializer=tf.zeros_initializer,
                dtype=tf.float32
            )
            
            relu_layer = tf.layers.dense(
                inputs=inputs, 
                units=end_units, 
                activation=tf.nn.relu, 
                name='dense_relu', 
                reuse=True
            )
            
            droprate = tf.placeholder(dtype=tf.float32, shape=(), name='droprate')
            #batch_norm = tf.contrib.layers.batch_norm(inputs=inputs, is_training=self.training, scale=True)
            
            self.feed_testing[droprate] = 1 # setting droprate for each layer
            self.feed_dropout[droprate] = 0.5
            #self.feed_full[droprate] = 1 # is not used any more
            self.feed_training[droprate] = 1
            
            return tf.layers.dropout(inputs=relu_layer, rate=droprate, training=self.training, seed=self.tf_seed)
    
    '''
    def generate_model(self): # returns reconstruction+other layers
        
        #sets up model by units, creates all necessary layers
        
        # setup tf variables for use in each layer, by shape. names given coorespond to layer, then biases and weights
        
        layers = {}
        
        layers['rescaled'] = self.rescale(self.X, is_training=self.training)
        
        previous_layer = self.dense_layer(inputs=layers['rescaled'], start_units=self.datas.interaction_data.shape[0], end_units=self.units[0], name='input')# first, and nonhidden layer
        layers['input'] = previous_layer

        for i in range(len(self.units)-1):
            name = 'hidden_'+str(i)
            previous_layer = self.dense_layer(inputs=previous_layer, start_units=self.units[i], end_units=self.units[i+1], name=name)
            layers[name] = previous_layer

        previous_layer = self.dense_layer(inputs=previous_layer, start_units=self.units[len(self.units)-1], end_units=self.datas.interaction_data.shape[0], name='output', activation=tf.nn.relu)# last, and nonhidden layer
        layers['output'] = previous_layer
        
        self.lm.model = layers # return original, normalized, hidden layer 1, hidden layer 2, then the reconstruction
    '''
    
    def setup_sess(self):
        #tf.reset_default_graph() # is this needed?
        #self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.paths.model_path)

    def pca_summary(self, Var='hidden_0'):
        if self.paths.is_recording:
            from tensorflow.contrib.tensorboard.plugins import projector
            embedding_writer = tf.summary.FileWriter(self.paths.get_model_path('embeddings'), self.sess.graph)
            
            #variables = [tf.Variable(initial_value=self.sess.run(self.model[key], feed_dict=self.feed_full), name=key) for key in self.model.keys()]
            lay = self.compressed_from_layer(Var)
            #np.save(self.model_folder+'/embeddings/layer_'+Var, lay)
            var = tf.Variable(lay, name=Var)
            
            projection_saver = tf.train.Saver([var])
            #self.sess.run([var.initializer for var in variables])
            self.sess.run(var.initializer)
            projection_saver.save(self.sess, self.paths.model_path('/embeddings/embeddings.ckpt'))
            
            config = projector.ProjectorConfig()
            
            def add_embedding(var):
                embedding = config.embeddings.add()
                print(var.name)
                embedding.tensor_name = var.name
                embedding.metadata_path = self.paths.metadata_labels
            
            #[add_embedding(var) for var in variables]
            add_embedding(var)
            
            projector.visualize_embeddings(embedding_writer, config)

    def epoch_summary(self):
        print(self.datas.training.epoch, self.datas.training.iteration)
        if self.paths.is_recording:
            summary = self.sess.run(self.tf_summary, feed_dict=self.feed_testing)
            self.summary_writer_testing.add_summary(summary, self.datas.training.epoch)
            #rescaled, output = self.sess.run([self.model['rescaled'], self.model['output']], feed_dict=self.feed_training)
            #print(rescaled.shape)
            #np.savetxt(self.model_folder+'rescaled_layer_'+str(self.datas.training.epoch), rescaled)
            #np.savetxt(self.model_folder+'output_layer_'+str(self.datas.training.epoch), output)
    
    def train_summary(self, data):
        if self.paths.is_recording:
            self.feed_training[self.X] = data
            summary = self.sess.run(self.tf_summary, feed_dict=self.feed_training)
            self.summary_writer_training.add_summary(summary, self.datas.training.iteration)
    
    '''def compressed(self): # needs to be redone and checked
        full_interactions = proteinLoader.proteinloader(self.interaction_data, 100, labels=self.label_data)
        self.feed_training[self.X] = full_interactions.next_batch()
        full = self.sess.run(self.model['hidden_0'], feed_dict=self.feed_training)
        while full_interactions.epoch==0:
            self.feed_training[self.X] = full_interactions.next_batch()
            batch_compressed = self.sess.run(self.model['hidden_0'], feed_dict=self.feed_training)
            full = np.vstack([full, batch_compressed])
        return full[:19576]
    
    def compressed_from_layer(self, layer_name): # needs to be redone and checked
        full_interactions = proteinLoader.proteinloader(self.interaction_data, 100, labels=self.label_data)
        self.feed_training[self.X] = full_interactions.next_batch()
        full = self.sess.run(self.model[layer_name], feed_dict=self.feed_training)
        while full_interactions.epoch==0:
            self.feed_training[self.X] = full_interactions.next_batch()
            batch_compressed = self.sess.run(self.model[layer_name], feed_dict=self.feed_training)
            full = np.vstack([full, batch_compressed])
        return full[:19576]'''
    
    def compressed(self):
        batcher = self.datas.training
        self.feed_training[self.X] = self.datas.testing_data
        full = self.sess.run(self.model['hidden_0'], feed_dict=self.feed_training)
        batcher.reset()
        while batcher.epoch==0:
            self.feed_training[self.X] = batcher.next_batch()
            batch_compressed = self.sess.run(self.model['hidden_0'], feed_dict=self.feed_training)
            full = np.concatenate([full, batch_compressed])
        return full[:19576]
    
    def train_mse(self, epochs=5):
        self.epoch_summary()
        self.train_summary(self.datas.training.next_batch())
        self.datas.training.set_next_epoch(self.epoch_summary, ())
        while self.datas.training.epoch<=epochs:
            proteins = self.datas.training.next_batch()
            self.feed_dropout[self.X] = proteins
            self.sess.run(self.train_step_mse, feed_dict=self.feed_dropout)
            if not self.datas.training.iteration%20:
                if self.paths.is_recording:
                    self.train_summary(proteins)
                #print(self.datas.training.epoch, self.datas.training.iteration)
        if self.paths.is_recording:
            self.saver.save(self.sess, self.paths.model_path)
    
    def train_mix_fac_layers_sigmoid_crossentropy_loss(self, epochs=20):
        self.epoch_summary()
        self.train_summary(self.datas.training.next_batch())
        self.datas.training.set_next_epoch(self.epoch_summary, ())
        while self.datas.training.epoch<=epochs:
            proteins = self.datas.training.next_batch()
            self.feed_dropout[self.X] = proteins
            self.sess.run(self.train_step_sce_mix_fac, feed_dict=self.feed_dropout)
            if not self.datas.training.iteration%20:
                if self.paths.is_recording:
                    self.train_summary(proteins)
        if self.paths.is_recording:
            self.saver.save(self.sess, self.paths.model_path)
    
    def train_sigmoid_crossentropy_loss(self, epochs=30):
        self.epoch_summary()
        self.train_summary(self.datas.training.next_batch())
        self.datas.training.set_next_epoch(self.epoch_summary, ())
        while self.datas.training.epoch<=epochs:
            proteins = self.datas.training.next_batch()
            self.feed_dropout[self.X] = proteins
            self.sess.run(self.train_step_sce, feed_dict=self.feed_dropout)
            if not self.datas.training.iteration%20:
                if self.paths.is_recording:
                    self.train_summary(proteins)
        if self.paths.is_recording:
            self.saver.save(self.sess, self.paths.model_path)

if __name__=='__main__':
    
    import PathManager#.PathManager # makes code less messy, something i made to make code neeter
    import DataManager# another thing I made to make code less messy
    
    paths = PathManager.PathManager(
        is_recording=True
    )
    
    datas = DataManager.DataManager(
        paths,
        batch_size=20
    )
    
    ae = Autoencoder(paths, datas)
    
    #ae.train_mse(epochs=40)



