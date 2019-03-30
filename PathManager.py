# path manager
'''
holds all paths
keeps track of which files need to be saved, for each model
keeps track of output, and function folders
'''

from datetime import datetime
import subprocess
import os

class PathManager():
    
    def __init__(self, 
                 params, 
                 name=None, 
                 is_recording=True, 
                 ppi_path='data/PPI_normalized_3.npy', 
                 pnames  ='data/PPI-Order.npy', 
                 go_path ='data/GO.npy', 
                 go_func_order = 'data/Function-Order.npy', 
                 cancer_functions = 'data/CancerFunctions.npy', 
                 log='log.tsv', 
                 folder_format='%B/%d/%H_%M/'
                ):
        
        self.params = params
        
        self.join = os.path.join
        # Set path joining algorithm
        # to that of OS's.
        
        self.is_recording = is_recording
        # Determines whether data is written,
        # or not saved at all.
        
        if name==None:
            self.model_folder = self.params.get(
                'modeldir', 
                self.join(
                    'models', 
                    datetime.now().strftime(
                        self.params.get(
                            'folderformat', 
                            folder_format
                        )
                    )
                )
            )
            self.pretrained = False
        else:
            self.model_folder = self.params.get(
                'modeldir', 
                name
            )
            self.pretrained = True
        # The model folder is where all data is
        # stored. Here, I set it according to the
        # current time. Folder format determines
        # how folder of parented. Setting
        # folderformat to '%B/%d/%H_%M/' in params
        # would cause folder to be created in
        # in order of MONTH/DAY/TIME.
        
        
        self.params.set_save_dir(self.model_folder)
        # Change where params file will be saved.
        # Set it to this models folder, so when
        # params are written, they are saved to
        # the correct dir.
        
        if not self.pretrained:
            self.mkdirp(self.model_folder)
        # If this model is new, create the
        # needed folders, and their parent
        # directorys.
        
        self.model_path = self.params.get(
            'modelpath', 
            self.join(self.model_folder, 'model', 'model.ckpt')
        )
        # Set model path to that from within params.
        
        self.output = self.params.get(
            'ourputdir', 
            self.join(self.model_folder, 'output')
        )
        self.mkdirp(self.output)
        # Set output dir. This is where all
        # images, and other file are saved.
        # Also make the output folder.
        
        self.log = self.params.get(
            'logpath', 
            self.join(self.output, log)
        )
        # Get log file location.
        
        self.ppi = self.params.get(
            'ppipath', 
            ppi_path
        )
        # Get ppi file location.
        
        self.ProteinOrder = self.params.get(
            'proteinorder', 
            pnames
        )
        # Get ppi file protein order data location.
        
        self.FunctionOrder = self.params.get(
            'functionorder', 
            go_func_order
        )
        # Get order of functions in annotations.
        
        self.go = self.params.get(
            'gopath', 
            go_path
        )
        # Get GO file indexed by Gene location.
        
        self.CancerFunctions = self.params.get(
            'cancerfunctionspath', 
            cancer_functions
        )
        
        self.metadata_labels = self.params.get(
            'metadatapath', 
            '../../../../../data/tensorboard_labels_short.tsv'
        ) # must be relative because of tf
        # Get metadata file location.
        # This is used by Tensorboard.
        
        self.save_file(__file__)
    
    def save_file(self, path):
        
        if self.is_recording and not self.pretrained:
            subprocess.call(['cp', os.path.realpath(path), self.model_folder])
        # Save this file in model folder if it is new, and recording.
    
    def mkdirp(self, d):
        if self.is_recording:
            subprocess.call(['mkdir', '-p', d])
        # Useful function for making directory trees.
