import numpy as np

class KNN():
    
    def __init__(self, 
                 paths, 
                 params, 
                 GO, 
                 X, 
                 neighbor_count=1000
                ):
        
        self.params = params
        
        paths.save_file(__file__)
        # Save the current file to this
        # models freshly created directory.
        
        self.X = X
        # X is the PPI data,
        # but can be
        # compreseded in any
        # way.
        
        self.GO = GO
        # Set Gene Ontology
        # handler.
        
        self.neighbor_count = self.params.get('k-KNN', neighbor_count)
        # Number of nearest neighbors to look at.
    
    def predict(self):
        '''
        Make all predictions for all members
        of X.
        '''
        
        nn = self.find_nearest(
            self.X, 
            self.neighbor_count
        )[0].astype(np.int16)
        # Find the nearest k neighbors in X
        
        
        
        '''for nearest_neighbors in self.nearest_neighbors:
            self.preds.append(
                self.average_neighbors(nearest_neighbors)
            )'''
        weights = 1.0/(np.arange(self.neighbor_count)+1)
        weights *= 1.0/(weights.sum()+1.0/1000)
        weights = weights.astype(np.float32)
        # Make a falloff to weight known neighbors 
        # based off of their rank distance from
        # the current protein.
        # Cast to float32, because this was much
        # faster than float64. GO data is also
        # float 32, even though it is binary data.
        
        preds = np.zeros(shape=(self.X.shape[0], self.GO.data.shape[1]), dtype=np.float32)
        # Allocate an array where predictions will be stored. 
        # This array is proteins X functions
        
        
        
        '''#labeled = self.GO.data[nn[i]]#.astype(np.float32)
        # Get next proteins annotations
            
        #m = np.any(labeled<0, axis=1)
        # Calculate mask. All numbers<0
        # are now 1. Used to set all
        # -1's to 0.
            
        #label_weights = np.copy(weights)
        #label_weights[m]=0.0
        # Mask out all -1 using
        # weights.'''
        
        for i in range(nn.shape[0]):
            preds[i]+= np.dot(weights, self.GO.data[nn[i]])
        # Average the nearest neighbors of a protein
        # to make function predictions.
        
        #preds = np.asarray([np.dot(weights, self.GO.data[n]) for n in nn]
        
        return preds
    
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
