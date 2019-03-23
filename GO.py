import numpy as np

class GO():
    
    def __init__(self, paths, params):
        
        paths.save_file(__file__)
        # Save the current file to this
        # models freshly created directory.
        
        self.data = np.load(paths.go).astype(np.float32)
        # Loades in function annotations
        
        self.ProteinOrder = list(np.load(paths.ProteinOrder))
        # Loades list of proteins in order of protein
        
        self.FunctionOrder = list(np.load(paths.FunctionOrder))
        # Loades list of functions in order on annotations
        
        self.CancerFunctions = np.load(paths.CancerFunctions)
        
        #print(self.data.sum(), self.data.max())
        
    
    def get_protein(self, name):
        '''
        Gets a protein's annotations for all functions.
        Returns list of functions, with 1 indicating "done by protein"
        and 0 indicating "not done by protein".
        '''
        return self.data[self.ProteinOrder.index(name)]
    
    def get_function(self, func):
        '''
        Gets a list of proteins, with 1 indicating "does function"
        and 0 indicating "doesn't do function".
        '''
        return self.data.T[self.FunctionOrder.index(func)]

if __name__=='__main__':
    from PathManager import PathManager as PM
    from ParamHandler import ParamHandler as PH
    ph = PH()
    pm = PM(ph)
    go = GO(pm, ph)
    