
import numpy as np

class Log():
    
    def __init__(self, paths, cols=['Function', 'AUCROC', 'AUCPR'], sep='\t'):
        
        self.paths = paths
        self.paths.save_file(__file__)
        self.file = open(paths.log, 'a')
        self.sep = str(sep)
        self.append(cols)
    
    def __del__(self):
        '''
        Make sure to deallocate file.
        '''
        self.file.close()
    
    def append(self, elements):
        '''
        Append next elements to log,
        using correct seperators.
        '''
        if self.paths.is_recording:
            line = ''
            for e in elements:
                line+=str(e)+self.sep
            line = line[:-1]+'\n'
            self.file.write(line)
            self.file.flush()