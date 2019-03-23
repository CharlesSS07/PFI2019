import numpy as np
from sklearn import decomposition

class PPI():
    
    def __init__(self, 
                 paths, 
                 params, 
                 test_points=576, 
                 seed=0, 
                 batch_size=50
                ):
        
        self.params = params
        
        paths.save_file(__file__)
        # Save the current file to this
        # models freshly created directory.
        
        if seed == None:
            self.params.get('npseed-InteractionData', np.random.get_state()[1][0])
        else:
            np.random.seed(self.params.get('npseed-InteractionData', seed))
        self.batch_size = self.params.get('batchsize-InteractionData', batch_size)
        
        self.data = np.load(paths.ppi)
        self.pnames = np.load(paths.ProteinOrder)
        
        self.batch_length = self.data.shape[1]
        self.batches = self.data.shape[0]
        
        batches = np.arange(0, self.data.shape[0], dtype=np.int16)
        np.random.shuffle(batches)
        
        self.testing_batches = batches[:test_points]
        
        self.training_points = np.copy(batches[test_points:])
        self.training_batches = list(batches[test_points:])
        
        self.epoch = 0
        self.iteration = 0
        
        self.median = np.median(self.data)
        self.std = np.std(self.data)
        self.minimum = np.min((self.data-self.median)/self.std)
        
        #print(self.data.shape, len(self.training_batches), len(self.testing_batches), self.data.dtype)
    
    def next_batch(self):
        '''
        Returns list of protein names,
        protein interacitons, and
        indexes within PPI data.
        '''
        if len(self.training_batches)<self.batch_size:
            self.reload_batches()
            self.epoch+=1
            self.on_epoch()
        idx = [self.training_batches.pop() for _ in range(self.batch_size)]
        self.iteration += self.batch_size
        return self.pnames[idx], self.data[idx], idx
    
    def pca(self):
        '''
        Does PCA using SKlearn package.
        PCA looks across all dimensions,
        and keeps only dimensions of
        highest variance.
        
        Turner off an on by pca-InteractionData in
        params file.
        Set number of components using
        pcac-InteractionData in params.
        '''
        if self.params.get('pca-InteractionData', True):
            pca = decomposition.PCA(
                n_components=self.params.get('pcac-InteractionData', 1000), 
                svd_solver='randomized'
            )
            pca.fit(self.data)
            self.data = pca.transform(self.data)
            self.batch_length = self.data.shape[1]
    
    def on_epoch(self):
        '''
        Run once at beginning of every epoch.
        Overide to have epochly event.
        '''
        print('Epoch:', self.epoch)
    
    def reload_batches(self):
        '''
        Run to reset epoch.
        '''
        self.training_batches =  list(np.copy(self.training_points))
    
    def get_testing(self):
        '''
        Returns data singled out for testing.
        This is only way to get testing data.
        '''
        return self.data[self.testing_batches]

if __name__=='__main__':
    from PathManager import PathManager as PM
    from ParamHandler import ParamHandler as PH
    
    ph = PH()
    pm = PM(ph)
    
    ppi = PPI(pm, ph)
    ppi.pca()
    print(ppi.data.shape, len(ppi.training_batches), len(ppi.testing_batches))