#import yaml
from ruamel import yaml

class ParamHandler():
    
    def __init__(self, file='params.yaml'):
        
        self.magic_vals = {}
        self.file = file
        
    def get(self, name, val):
        
        if type(name)!=str:
            raise ValueError('Variable name must be a string!')
        
        if name not in self.magic_vals:
            self.magic_vals[name] = val
        
        self.on_get(name, self.magic_vals[name])
        
        return self.magic_vals.get(name)
    
    def on_get(self, name, val):
        print(name+'\t', val)
    
    def set_save_dir(self, path, file='params.yaml'):
        
        self.file = path+file
    
    def save(self):
        
        with open(self.file, 'w') as f:
            f.write(yaml.dump(self.magic_vals))
    
    def load(self):
        
        try:
            with open(self.file, 'r') as f:
                self.magic_vals = yaml.load(f)
        except FileNotFoundError:
            self.magic_vals = {}

if __name__=='__main__':
    
    params = ParamHandler(file='/tmp/foo.yaml')
    
    a = params.get('a', 17.5)
    
    b = params.get('b', 'test')
    
    #r = params.get(['c', 'b'], 'r') # should erroe because name must be str
    
    print(a, b)
    
    params.save()
    
    del params
    
    params = ParamHandler(file='/tmp/foo.yaml')
    
    params.load()
    
    a = params.get('a', 21.5)
    
    b = params.get('b', 'test2')
    
    #r = params.get(['c', 'b'], 'r') # should not load because name must be str, and was never saved
    
    print(a, b)