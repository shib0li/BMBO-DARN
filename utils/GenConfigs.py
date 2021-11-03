
class GenerateFcnnConfig(object):
    
    domain_name = None
    placement = None
    sobol = None
    N = None
    seed = None
    horizon = None
    

    def _parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        print('Generation Config:')
        print('=======================')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, ':', getattr(self, k))
        print('=======================')