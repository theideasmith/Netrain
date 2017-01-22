import numpy as np
def dkey(d, key):                                                         
  return d[key] if key in d else None                                     
def defaultifnone(var, df):                                               
  return var if var != None else df                                       
def randint(start=None, end=None, size=None):                             
  start = defaultifnone(start, 1)                                         
  end = defaultifnone(end, 100)                                           
  size = defaultifnone(size, None)                                        
  return np.random.random_integers(start, high=end, size=size)            
  
def randfloat(start=None, end=None, size=None):
  start = defaultifnone(start, 1)                                          
  end = defaultifnone(end, 100)                                            
  size = defaultifnone(size, 1)                                         
  return np.random.random(size=size)* ( end-start) + start                                                                                      

def randbool(size=None):                                                  
  size = defaultifnone(size, 1)                                           
  return np.random.choice([0,1], size)                                    
                                                                          
def randchoice(array, size=None):                                         
  size = defaultifnone(size, 1)                                           
  return np.random.random_choice(array, size)                             

def randint(start=None, end=None, size=None):                              
  start = defaultifnone(start, 1)                                          
  end = defaultifnone(end, 100)                                            
  size = defaultifnone(size, None)                                         
  return np.random.randint(start, high=end, size=size)             

def paramGen(param):                                                      
   pp = param                                                             
   kind = param['type']                                                   
   if kind in {"int", "float"}:                                                      
     start = dkey(pp, 'start')                                            
     end = dkey(pp, 'end')                                                
     size = dkey(pp, 'size')                                              
     if kind == "int": 
       rand = randint(start=start, end=end, size=size)                      
     if kind =="float":
       print "Gen Float"
       rand = randfloat(start=start, end=end, size=size)
     return rand                                                          
   elif kind=='bool':                                                     
     return randbool(size=size)                                           
   elif kind =='choice':                                                  
     array = pp['array']                                                  
     return randchoice(array, size=size)                                  
                                                                          
def parameterGen(params):                                                 
  results ={}                                                             
  for key in params:                                                      
    pp = param_params = params[key]                                       
    kind = pp['type']                                                     
    if not kind in ['int', 'bool', 'choice']:                             
      raise Error('Requested parameter type not supported')               
                                                                          
    size = dkey(pp, 'size')                                               
    # Enabling size itself to be dependent on another parameter           
    if size in params:                                                    
      if params[size]['type'] != 'int':                                   
        raise Error('Cannot assign noninteger random size')               
      if not size in results:                                             
        results[size] = paramGen(size[size])                              
      params[key]['size'] = results[size]                                 
    results[key] = paramGen(params[key])                                  
  return results                                                          

std_ann_params = {                                 
  'layer_sizes': {'type':'int', 'size':'layers'} , 
  'layers': {'type': 'int', 'start': 1, 'end':10}  
}                                                  
