import numpy as np
#import csv
from sklearn import decomposition
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import subprocess
import os

#import proteinLoader # a package I made for loading the protein data
#from autoencoder_ss_v3 import Autoencoder # imports all of my auto encoder
#from protein_label import ProteinLabelManager

class KNN():
    
    def __init__(self, paths, datas, ae=None, neighbor_count=1000):
        
        self.paths = paths
        self.paths.save_file(__file__)
        
        self.datas = datas
        
        self.ae = ae
        
        if self.ae==None: # X could be compressed autoencoeder latent layer, or just pure pca
            self.X = self.datas.interaction_data
        else:
            self.X = self.ae.compressed() # function still needs to be correcly implemented
        
        self.labels = self.datas.labels
        
        self.threshold_samples = np.arange(1, 0, -0.001)*1.1-0.05 # rescaled by *1.1-0.05 in order to make this overlap guesses, helped alot to make threhsolds be relevant. without this, output would not extend all the way to ends of plot causeing auc to be wrong
        
        #print(self.threshold_samples)
        self.neighbor_count = neighbor_count
    
    def predict(self, components):
        
        self.do_pca(components)
        #print(self.Xp.shape, self.Xp[1:6])
        self.nearest_neighbors = self.find_nearest(self.Xp, self.neighbor_count)[0]
        #print(self.nearest_neighbors.shape, self.nearest_neighbors[1:6])
        #print(self.nearest_neighbors, self.nearest_neighbors.shape, self.nearest_neighbors.min())
        
        preds = np.asarray([self.average_neighbors(nearest_neighbor) for nearest_neighbor in self.nearest_neighbors])
        preds_new = np.asarray([self.average_neighbors_new(nearest_neighbor) for nearest_neighbor in self.nearest_neighbors])
        
        aa = np.sum(preds>0.1,axis=0)
        aa_new = np.sum(preds_new>0.1,axis=0)
        for i in zip(aa,aa_new):
            print(i)
        #print(preds.shape, preds[1:6])
        #print(self.datas.label_data[self.nearest_neighbors[1]])
        #print(self.datas.label_data[self.nearest_neighbors[3]])
        #del self.datas.label_data
        self.labels.save_guesses(preds)
        #print(self.guesses.shape, self.guesses)
        return preds
    
    def do_pca(self, components):
        print(self.X.shape)
        pca = decomposition.PCA(n_components=components, svd_solver='randomized')
        pca.fit(self.X)
        self.Xp = pca.transform(self.X)
    def average_neighbors_new(self, neighbors):
        
        #print('neighbor', neighbors.shape, neighbors)
        labeled = self.datas.label_data[neighbors]
        m = np.any(labeled<0, axis=1)
        
        print('a:', labeled.shape, labeled.min(), (labeled<0).sum()) #np.asarray(self.datas.label_data[n] for n in neighbors])
        
        weights = 1.0/(np.arange(neighbors.size)+1) # how much a point is worth based off of its nearest neighbors
        weights[m]=0.0
        weights *= 1.0/(weights.sum()+1.0/1000)
        
        
        
        pred = np.dot(weights, labeled)
        
        print('b:', pred.shape, pred.min(), (pred<0).sum())
        #pred = np.sum(labeled*weights, axis=1)
        #print('pred:', pred[:10])
        #print(pred.shape) # pred is threshold for recognition shaped [N_Genes, N_functions]
        
        return pred#np.asarray([np.greater(pred, threshold).astype(int) for threshold in self.threshold_samples])
    def average_neighbors_old(self, neighbors):
        
        #print('neighbor', neighbors.shape, neighbors)
        labeled = self.datas.label_data[neighbors]
        m = np.any(labeled<0, axis=1)
        
        print('a:', labeled.shape, labeled.min(), (labeled<0).sum()) #np.asarray(self.datas.label_data[n] for n in neighbors])
        
        weights = 1.0/(np.arange(neighbors.size)+1) # how much a point is worth based off of its nearest neighbors
        #weights[m]=0.0
        weights *= 1.0/(weights.sum()+1.0/1000)
        
        
        
        pred = np.dot(weights, labeled)
        
        print('b:', pred.shape, pred.min(), (pred<0).sum())
        #pred = np.sum(labeled*weights, axis=1)
        #print('pred:', pred[:10])
        #print(pred.shape) # pred is threshold for recognition shaped [N_Genes, N_functions]
        
        return pred#np.asarray([np.greater(pred, threshold).astype(int) for threshold in self.threshold_samples])
    average_neighbors = average_neighbors_new
    def get_precision_recall_fscore_support(self, y_true, y_pred):
        y_true = np.greater(y_true, 0).astype(int)# using ints makes things much faster, should be ints anyway
        #y_pred = np.greater(y_pred, 0).astype(int)
        #print('totals predy and truey:', y_pred.sum(), y_true.sum())
        precision, recall, fbscore, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
        return precision, recall, fbscore, support
    
    def fpr_tpr(self, y_true, y_pred): # must be binary
        y_true = np.greater(y_true, 0).astype(int)# using ints makes things much faster, should be ints anyway
        y_pred = np.greater(y_pred, 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return (fp/(fp+tn)), (tp/(tp+fn))
    
    def select(self, guess):
        '''
        selects 1 or 0 for each index in guess, based on threshold
        '''
        mn = guess.min()
        rng = guess.max()-mn
        return np.asarray([guess>=threshold*rng+mn for threshold in self.threshold_samples])
    
    def thresholded_get_precision_recall_fscore_support(self, data, guess):
        
        mn = guess.min()
        rng = guess.max()-mn# mn and rng for rescaleing thresholds according to guesses, makes thresholds be compairable to guess
        return np.transpose(
            [self.get_precision_recall_fscore_support(data, guess>=threshold*rng+mn) for threshold in self.threshold_samples], 
            axes=(1, 2, 0)
        )
    
    
    
    def graph_collum(self, label):
        
        print(label.data, label.data.shape)
        
        known = label.data!=-1
        if known.sum()==0:
            print('nothing has function:', label.Funciton)
            return None
        data = label.data[known]
        guess = label.guess[known]
        #anyunkn = np.any(unknown)
        #allunkn = np.all(unknown)
        
        mn = guess.min()
        rng = guess.max()-mn# mn and rng for rescaleing thresholds according to guesses, makes thresholds be compairable to guess
        
        precision, recall, fbeta, support = self.thresholded_get_precision_recall_fscore_support(data, guess)
        
        fpr, tpr = np.transpose(
            [self.fpr_tpr(data, guess>=threshold*rng+mn) for threshold in self.threshold_samples], 
            axes=(1, 0))
        x = self.threshold_samples
        
   #     self.graph_pr(precision, recall, fbeta, x)
        
   # def graph_pr(self, precision, recall, fbeta, x):
        
        fig1, (rocplt, pcrplt) = plt.subplots(2, 1)
        
        #plotting
        pcrplt.plot(x, precision[1])
        pcrplt.plot(x, recall[1])
        pcrplt.plot(x, fbeta[1])
        pcrplt.set_xlim(0, 1)
        pcrplt.set_ylim(0, 1)
        pcrplt.set_xlabel('Threshold')
        pcrplt.set_ylabel('%')
        pcrplt.legend(['y = precision on 1\'s',
                       'y = recall on 1\'s',
                       'y = fbeta on 1\'s',], 
                      loc='lower right')
        
        #precision0 = np.concatenate(([[precision[0][0]], precision[0], [precision[0][-1]]]))
        #precision1 = np.concatenate(([[precision[1][0]], precision[1], [precision[1][-1]]]))
        #print(precision[1][0], precision[1][-1], precision)
        rocplt.plot(np.concatenate(([0], fpr, [1])), np.concatenate(([tpr[0]], tpr, [tpr[-1]])))
        rocplt.plot(np.concatenate(([0], tpr, [1])), np.concatenate(([[precision[1][0]], precision[1], [precision[1][-1]]])))
        
        rocplt.set_title('roc')
        rocplt.set_xlim(0, 1)
        rocplt.set_ylim(0, 1)
        rocplt.set_xlabel('FPR,Recall')
        rocplt.set_ylabel('TPR,Precision')
        rocplt.legend(['fpr and tpr',
                       'recall, precision on 1\'s'], 
                      loc='lower center')
        
        #plt.show()
        label.save_plot(fig1)
        
        #label.save_metrics(precision, recall, fbeta, tpr, fpr) # deprediated
        
        #get auc's
        aucroc = np.trapz(np.concatenate(([tpr[0]], tpr, [tpr[-1]])), np.concatenate(([0], fpr, [1])))
        
        aucpr0 = np.trapz(np.concatenate(([precision[0][0]], precision[0], [precision[0][-1]])), np.concatenate(([0], tpr, [1])))
        aucpr1 = np.trapz(np.concatenate(([precision[1][0]], precision[1], [precision[1][-1]])), np.concatenate(([0], tpr, [1])))
        print(mn, rng)
        label.write(aucroc, aucpr0, aucpr1, precision[1], recall[1])
        print('Function:', label.function)
        print('AUC_ROC: ', aucroc)# tpr and fpr are backwards, so must extrapolate from 0'th element
        print('AUC_PR0: ', aucpr0)
        print('AUC_PR1: ', aucpr1)
        print('All Known', known.sum())
        #print('Proteins: ', self.datas.label_data.shape[0], support, 'Percent 1\'s:', support[1]/self.datas.label_data.shape[0])
        plt.close()
    
    def graph_collums(self):
        
        [self.graph_collum(label) for label in self.labels.labels]
    
    def find_nearest(self, yy, topK=10):
        # extract the nearest topK points to every point
        # exlcuidng self as being one of the nearest.

        # to do this efficiently in time and space in numby
        # use trick:
        #  dist^2 = (y-x)^2 = y^2 +x^2 - 2*x*y
        # you can use vector methods on all the last 3 terms

        vv = np.sum(yy*yy,axis=1)
        vv = (vv[:,np.newaxis]+vv.T)-np.dot(2*yy,yy.T)

        # this is trick to remove the self cases
        irx = np.arange(vv.shape[0])
        temp = np.max(vv)+1
        vv[irx,irx] = temp  # stuff diagonals 
        
        # find the best in every row
        ss = np.argpartition(vv, kth=topK)[:,:topK]
        #print(ss.shape, ss.max())
        
        # extract the values for the topK
        jj= vv[np.arange(ss.shape[0])[:,np.newaxis],ss]
        del vv
        
        # now we can sort these topK into rank order by any criteria
        # this order does not have to be the distance!!
        # but here we will use distance to sort these closes topK points.
        
        # these topK are not yet sorted.
        sj = np.argsort(jj,axis=1)  
        #print(sj.shape, sj.max())
        # these index refer to the truncated set
        # to link this back to the original data we apply this arg sort to the previously
        # argpartitioned indicies
        #vf = np.asarray([ss[i,j] for i,j in enumerate(sj)]) # uses list comprehension
        vf = ss[irx[:,np.newaxis],sj]  # same thing but numpy fancy indexing
        mf = jj[irx[:,np.newaxis],sj]
        #print(vf.max(), mf.max())
        return vf, mf

if __name__=='__main__':
    
    import PathManager#.PathManager # makes code less messy, something i made to make code neeter
    import DataManager# another thing I made to make code less messy
    
    paths = PathManager.PathManager(
        name='models/January/25/22_08',
        is_recording=True
    )
    
    datas = DataManager.DataManager(
        paths,
        batch_size=5,
        stage='test'
    )
    
    knn = KNN(paths, datas)
    knn.predict(components=2000)
    
    knn.graph_collums()



