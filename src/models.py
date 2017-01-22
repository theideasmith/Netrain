import os
import Constants as const

os.path.append( const.PROJECT_DIR )
                                                                                                   
def anetwork_parameters(initial_params):                                                          
   architecture = ifnone(initial_params,'architecture', const.DEFAULT_ARCHITECTURE)                 
   nb_epoch = ifnone(initial_params,'nb_epoch', const.MCARLO_NB_EPOCH)                              
   epoch_samples = ifnone(initial_params,'epoch_samples', const.MCARLO_EPOCH_SAMPLES)               
                                                                                                    
   archpath = ifnone(initial_params,'archpath', const.ARCHITECTURE_PATH)                            
   weightpath = ifnone(initial_params,'wpath', const.WEIGHTS_PATH)                                  
   metricspath = ifnone(initial_params,'metricspath', const.MODEL_METRICS_PATH)                     
   model_generator = archgen['classifier'][initial_params["architecture"]]                    
   training_generator = nnets.traingen['classifier']                                                
   evaluation_generator = nnets.evalgen['classifier']                                               

   config = {                                                                                       
     "name": "classifier_model",                                                                    
     "nb_epoch": nb_epoch,                                                                          
     "epoch_samples": epoch_samples,                                                                
     "archpath": archpath,                                                                          
     "wpath": weightpath,                                                                           
     "metricspath":metricspath,                                                                     
     "model_generator": model_generator,                                                            
     "training_generator": training_generator,                                                      
     "evaluation_generator": evaluation_generator,                                                  
     "optimizer": nnets.modeldata["classifier"][initial_params["architecture"]]["optimizer"],       
     "loss": nnets.modeldata["classifier"][initial_params["architecture"]]["loss"]                  
   }                                                                                                
   return config                                                                                    
                                                                                                    
