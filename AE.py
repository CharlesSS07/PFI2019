import numpy as np
import tensorflow as tf

class AE():
    
    def __init__(self, 
                 paths, 
                 params, 
                 PPI, 
                 learn_rate=0.001
                ):
        
        self.params = params
        # Set the paramater handler.
        
        self.paths = paths
        self.paths.save_file(__file__)
        # Save the current file to this
        # models freshly created directory.
        
        self.seed = self.params.get('npseed-AE', 1)
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        # Set all seeds for testing purposes.
        # Random seeds makes results vary.
        
        
        self.PPI = PPI
        # The PPI dataset manager.
        
        #gpu_options = tf.GPUOptions(allow_growth=True)
        # Allows for dynamic memory allocation on GPU.
        
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()
        # Instanciates a tf Session with special gpu options to do dynamic memory.
        # By default, TF uses all avaliable memory, which makes running multiple
        # models impossible. Using dynamic memory allocation, debugging becomes
        # simpler because the number of allocated bytes can be seen changing over
        # time.
        
        self.learn_rate = self.params.get(
            'learnrate-AE', 
            learn_rate
        )
        
        self.X = tf.placeholder(
            dtype=tf.float32, 
            shape=(None, self.PPI.batch_length), 
            name='x'
        )
        # X is the input variable for tensorflow.
        # Later on, X will be repalced by a batch.
        
        self.is_training = tf.placeholder(
            dtype=tf.bool, 
            shape=(), 
            name='training'
        )
        # A variable indicating to TF whether or not
        # the model is being trained or testing.
        
        self.testing = {
            self.is_training: False, 
            self.X: self.PPI.get_testing()
        }
        
        self.training = {
            self.is_training: False, 
            self.X: self.PPI.next_batch()
        }
        
        self.dropout = {
            self.is_training: True, 
            self.X: self.PPI.next_batch()
        }
        # The following (testing, training, and dropout)
        # are inputs to the model. Each is slightly
        # different. Testing holds only data for testing
        # Training holde only data for training on.
        # Dropout is like training, but with dropouts
        # enabled.
        
        rescaled = self.min_max(self.rescale(self.X))
        
        self.input = self.dense_relu_layer(
            rescaled, 
            self.params.get('inputlayersize-AE', 4000), 
            'input'
        )
        self.latent = self.dense_relu_layer(
            self.input, 
            self.params.get('latentlayersize-AE', 2000), 
            'latent'
        )
        self.output = self.min_max(
            self.dense_relu_layer(
                self.dense_relu_layer(
                    self.latent, 
                    self.params.get('hiddenlayersize-AE', 4000), 
                    'hidden'
                ), 
                self.PPI.batch_length, 
                'output'
            )
        )
        # Sets up TF layers:
        #  rescaled - 19576, normalized so median=0
        #  input - 19576>>4000
        #  latent - 4000>>2000
        #  hidden - 2000<<4000
        #  output - 4000<<19576
        #
        # Dense_relu_layer is a function, which
        # correctly sets up and names a TF Layer.
        # 
        # Rescale is a function that rescales the
        # data to prevent the loss from fluctuating
        # due to the inputs having different scales.
        
        
        self.loss_meansquared = tf.losses.mean_squared_error(
            rescaled, 
            self.output
        )
        self.loss_crossentropy = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=rescaled,
                logits=self.output
            )
        )
        # Two different loss curves for evaluating loss on.
        # Record both, while training on a single. The loss
        # trained on varies by run. When trainig with cross
        # entropy loss, mse tells how far of each dimension
        # was on average, so we can tell how optimization is
        # going from two different perspectives.
        
        
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
        # Use adam optimizer because it adaptively changes learning rate.
        
        opt = self.params.get('optimizer-AE', 'SCE')
        if opt=='MSE':
            self.train_step = self.optimizer.minimize(loss=self.loss_meansquared)
        elif opt=='SCE':
            self.train_step = self.optimizer.minimize(loss=self.loss_crossentropy)
        else:
            raise ValueError('Value of "optimizer-AE" must be "MSE", or "SCE" in params file.')
        # Set optimizer for meansquared error or cross entropy loss.
        # Change optimizer-AE in params for MSE or SCE
        
        if self.paths.is_recording:
            self.summary_writer_training = tf.summary.FileWriter(
                paths.join(self.paths.model_folder, 'training'), 
                self.sess.graph
            )
            self.summary_writer_testing = tf.summary.FileWriter(
                paths.join(self.paths.model_folder, 'testing'), 
                self.sess.graph
            )
        else:
            self.summary_writer_training = tf.summary.FileWriter(
                paths.join('/tmp/', self.paths.model_folder, 'training'), 
                self.sess.graph
            )
            self.summary_writer_testing = tf.summary.FileWriter(
                paths.join('/tmp/', self.paths.model_folder, 'testing'), 
                self.sess.graph
            )
        # Two TF Summary wrighters that are later used to save the losses
        # to a graph format in Tensorboard.
        
        self.tf_summary = tf.summary.merge(
            [
                tf.summary.scalar('Cross-Entropy-Loss', self.loss_crossentropy), 
                tf.summary.scalar('Mean-Squared-Error', self.loss_meansquared)
            ]
        )
        # Defines a TF summary, that can be written out
        # to tensorboard for doing visualizations.
        
        '''
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        md = tf.RunMetadata()
        self.summary_writer_testing.add_run_metadata(md, 'test', 0)
        self.sess.run(tf.global_variables_initializer())#, options=run_options, run_metadata=md)
        self.summary_writer_testing.add_run_metadata(md, 'test', 1)
        '''
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # Set up context, initialize all vars, then set up saver.
        
        self.PPI.on_epoch = self.epoch_summary
        
        if not self.paths.pretrained:
            self.saver.restore(self.sess, self.paths.model_path)
        # Use saver to restore a model by name.
    
    def rescale(self, inputs, name='rescale'):
        '''
        Rescales a batch of data to have median of 0. Prevents loss from
        fluctuating because a batch has a high magnitude, offsets, or outliers.
        '''
        
        tf_MEDIAN = tf.constant(self.PPI.median,  dtype=tf.float32, name="MEDIAN")
        tf_STD = tf.constant(self.PPI.std,     dtype=tf.float32, name="STD")
        tf_MIN = tf.constant(self.PPI.minimum, dtype=tf.float32, name="MIN")
        
        with tf.variable_scope(name):
            if self.params.get(name+'layerdorescale-AE', True):
                return tf.subtract(tf.divide(tf.subtract(inputs, tf_MEDIAN), tf_STD),  tf_MIN)
            else:
                return inputs
    
    def min_max(self, inputs, name='min_max'):
        '''
        Normalize the vector between 0 and 1.
        '''
        with tf.variable_scope(name, dtype=tf.float32):
            mx = tf.reduce_max(inputs)
            mn = tf.reduce_min(inputs)
            
            rng = mx-mn
            
            return (inputs-mn)/rng
    
    def dense_relu_layer(self, inputs, end_units, name):
        '''
        Neat way of initializing layers. Automatically sets up vars/names, and more readable.
        Also creates variables for tensorboard, TF's graphing software.
        '''
        
        start_units = int(inputs.shape[1])
        
        with tf.variable_scope(name, dtype=tf.float32):
            
            if self.params.get(name+'layerdobatchnorm-AE', False):
                inputs = tf.contrib.layers.batch_norm(
                    inputs=inputs, 
                    is_training=self.is_training, 
                    scale=self.params.get(
                        name+'layerbatchnormdoscale-AE', 
                        True
                    )
                )
            # Adds a layer of batch normalization to this layer,
            # when params have <name>layerdo_batchnorm-AE equal
            # to True.
            
            tf.get_variable(
                name='layer/kernel', 
                shape=(start_units, end_units), 
                initializer=tf.initializers.random_uniform, 
                dtype=tf.float32
            )
            # Create initializer for weights of layer.
            
            tf.get_variable(
                name='layer/bias', 
                shape=end_units, 
                initializer=tf.initializers.random_uniform, 
                dtype=tf.float32
            )
            # Create initializer for biases of layer.
            
            layer = tf.layers.dense(
                inputs=inputs, 
                units=end_units, 
                activation=tf.nn.relu, 
                name='layer', 
                reuse=True
            )
            
            # Create TF layer, with
            # relu activation.
            
            droprate = tf.placeholder(dtype=tf.float32, shape=(), name='droprate')
            
            self.testing[droprate] = 1
            self.training[droprate] = 1
            self.dropout[droprate] = 0.5#self.params.get(name+'layerdroprate-AE', 0.5)
            # Set what to fill in droprate with
            # later on.
            
            return tf.layers.dropout(
                inputs=layer, 
                rate=droprate, 
                training=self.is_training, 
                seed=self.seed
            )
            # Do dropouts on layer,
            # according to is_training
            # and dropout rate.
    
    def pca_summary(self, lay, name='latent'):
        if self.paths.is_recording:
            from tensorflow.contrib.tensorboard.plugins import projector
            embedding_writer = tf.summary.FileWriter(self.paths.join(self.paths.model_folder, 'embeddings'), self.sess.graph)
            
            var = tf.Variable(lay, name=name)
            
            projection_saver = tf.train.Saver([var])
            
            self.sess.run(var.initializer)
            projection_saver.save(self.sess, self.paths.join(self.paths.model_folder, 'embeddings/embeddings.ckpt'))
            
            config = projector.ProjectorConfig()
            
            embedding = config.embeddings.add()
            embedding.tensor_name = var.name
            embedding.metadata_path = self.paths.metadata_labels
            
            projector.visualize_embeddings(embedding_writer, config)
    
    
    def epoch_summary(self):
        print('Epoch:', self.PPI.epoch, self.sess.run(self.loss_meansquared, feed_dict=self.testing))
        '''
        Records a TF summary to Tensorboard. Uses test data, to get loss
        and plots loss over epochs.
        '''
        summary = self.sess.run(self.tf_summary, feed_dict=self.testing)
        self.summary_writer_testing.add_summary(summary, self.PPI.iteration)
    
    
    def train_summary(self, batch):
        '''
        Records a TF summary to Tensorboard. Uses train data, to get loss
        and plots loss over iteration.
        '''
        self.training[self.X] = batch
        summary = self.sess.run(self.tf_summary, feed_dict=self.training)
        self.summary_writer_training.add_summary(summary, self.PPI.iteration)
    
    def train(self, epochs=5):
        '''
        Trainings with dropouts.
        '''
        self.epoch_summary()
        self.train_summary(self.PPI.next_batch()[1])
        epochs = self.params.get('epochs-AE', epochs)
        while self.PPI.epoch < epochs:
            self.dropout[self.X] = self.PPI.next_batch()[1]
            self.sess.run(self.train_step, feed_dict=self.dropout)
            #if not self.PPI.iteration % 20:
            #    self.train_summary(self.dropout[self.X])
            self.train_summary(self.dropout[self.X])
    
    def compress(self):
        '''
        Dimensionally compressed one batch of data.
        '''
        c = []
        for p in self.PPI.data:
            self.training[self.X] = [p]
            c.append(self.sess.run(self.latent, feed_dict=self.training))
        return np.asarray(c)
    
    def save(self):
        '''
        Saves model for later use.
        '''
        self.saver.save(self.sess, self.paths.model_path)