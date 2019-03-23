import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pylab as pl

class Metrics():
    
    def __init__(self, paths, params, Y, GO):
        
        self.params = params
        self.paths = paths
        self.paths.save_file(__file__)
        # Set path ans params
        # managers
        
        self.Y = Y
        # Set predictions array
        
        self.GO = GO
        # Set label manager
        
        if self.Y.shape[0]!=self.GO.data.shape[0]:
            raise ValueError('Must have same number of predictions and labels.')
        
        self.thresholds = np.arange(
            self.params.get('thresholdmin-Metrics', 0.00),
            self.params.get('thresholdmax-Metrics', 1.01),
            self.params.get('thresholdincrement-Metrics', 0.01)
        )
        # Which thresholds to threshold Y with.
    
    def metric(self, y_true, y_pred):
        '''
        What algorithim to use when thresholding.
        '''
        return precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    
    def threshold(self, guess):
        '''
        Returns this guess array, thresholded
        by each value in thresholds.
        '''
        mn = guess.min()
        rng = guess.max()-mn
        # Vars mn and rng for rescaleing
        # thresholds according to guess,
        # makes thresholds be compairable
        # to guess.
        
        thresholds = (self.thresholds*rng)+mn
        # Rescale thresholds for this guess.
        
        return ((thresholds[:,np.newaxis]-guess)<=0).astype(np.int8)
        # Makes Array of thresholdsXguesses, with threshold minus
        # guesses at each point, then returns 1 where threshold was
        # less than guess. 
    
    def run(self, show=False):
        '''
        Thresholds each funciton annotation guess,
        then compaires using metric to known annotations.
        '''
        for f in self.GO.CancerFunctions:
            index = self.GO.FunctionOrder.index(f)
            annotation = self.GO.get_function(f)
            # Get annotation for this protein
            thresholded = self.threshold(self.Y.T[index])
            # Get thresholded predictions (ThresholdsXFunctionAnnotations)
            metrics = []
            for prediction in thresholded:
                metrics.append(self.metric(annotation, prediction))
            metrics = np.asarray(metrics)
            z = metrics[:,0]==0
            metrics[:,0][z] = 1
            pl.figure()
            pl.plot(metrics[:,1,1], metrics[:,0,1], label='1')
            pl.plot(metrics[:,1,0], metrics[:,0,0], label='0')
            #pl.plot(self.thresholds, metrics[:,2,1])
            pl.xlabel('Recall')
            pl.ylabel('Precision')
            pl.title('PR '+f)
            pl.xlim(0, 1)
            pl.ylim(0, 1)
            pl.legend()
            f = f[:2]+'-'+f[3:]
            pl.savefig(self.paths.join(self.paths.output, f+'.pdf'))
            if show:
                pl.show()
            pl.close()