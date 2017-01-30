from AutoEncoderArchitectures import *
import pprint
import math
import io, json
import Constants as const
from itertools import *
import keras
from Generators import * 
from os import path
from utilities import ifnone
import models as nnets
from itertools import *
"""
We can do a random sampling over both numerical and discrete variable:
    - Number of layers
    - Number of hidden nodes per layer
    - Activation functions being used
    - Type of layers, stacking patterns

However, given that training a network takes time, 
doing full hyperparameter optimization is not 
necessary. const.I have preexisting layers model generators
and we look at how the various models perform.
"""
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = 0
    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class ModelTrainer:
  def __init__(
      self,
      model,
      params
    ):
    self.name = params['name']
    self.model = params['model']
    self.evaluation_generator = params['evaluation_generator']()
    self.training_generator = params['training_generator']()
    self.epoch_samples= params['epoch_samples']
    self.nb_epoch = params['nb_epoch']
    self.customhistory = LossHistory()
    self.history= keras.callbacks.History()
    self.architecture_path = os.path.join(params['archpath'], params['name'])+'.json'
    self.weights_path = os.path.join(params['wpath'], params['name'])+'.h5'
    self.auto_stop = params['intelligent']
    self.optimizer=params['optimizer']
    self.loss=params['loss']
    self.metrics=[self.loss]
    self.metrics_path = os.path.join(params['metricspath'], params['name']) + '.npy'
  
  def collect(self):
    self.save_architecture()
    self.model.compile(
        optimizer=self.optimizer,
        loss= self.loss,
        metrics=self.metrics,
        callbacks=[
          keras.callbacks.ModelCheckpoint(
            self.weights_path, 
            monitor='val_loss', 
            verbose=0, 
            mode='auto')
        ] 
    )                           
    print self.weights_path
    self.run()                                   
    # so we dont need to retrain every time
    # we save the weights
    self.save_weights()                                                 
    self.evaluate(save=True)

  def generator_to_data(self, generator, size):
    listed = islice(generator, size)
    listed = list(listed)
    print 'Len listed'
    arrayed = np.array(listed)
    training = arrayed[:,0]
    output = arrayed[:,1]
    return training, output

  def run(self):
    callbacks=[]
    saver =  keras.callbacks.ModelCheckpoint( 
               self.weights_path,             
               monitor='val_loss',            
               verbose=0,                     
            #  save_best_only=True,           
               mode='auto')                   
    callbacks.append(saver)
     
   #if self.auto_stop:
   #  callbacks.append(keras.callbacks.EarlyStopping( 
   #      monitor=self.loss,           
   #      patience=30,                 
   #      verbose=0,                   
   #      mode='min'))                 
    #inp, target = self.generator_to_data(
    #    self.training_generator,
    #    self.epoch_samples)

   # hist = self.model.fit(
   #   inp, target,
   #   verbose=1,
   #   callbacks=callbacks,
   #   batch_size=int(self.epoch_samples*0.1),
   #   nb_epoch = self.nb_epoch)
    traingen = cycle(islice(self.training_generator, 90000))
    hist = self.model.fit_generator(
      traingen,                               
      samples_per_epoch=self.epoch_samples,
      nb_epoch=self.nb_epoch,
      verbose=1, 
      callbacks=callbacks)

  def evaluate(self,
      val_samples=100, 
      save=False):
    results = self.model.evaluate_generator(
        self.evaluation_generator, 
        val_samples, max_q_size=10)

    if save==True:
      tosave = np.array(results)
      np.save(self.metrics_path, tosave)
    return np.array(results)

  def save_weights(self):
    self.model.save_weights(
        self.weights_path, overwrite=True)

  def save_architecture(self):
    stringified = unicode(self.model.to_json())
    with io.open(self.architecture_path, 'w') as f:
      f.write(stringified)
      f.close()

def modelevaluate(model, val_samples=1000, evaluation_generator=None):
  if evaluation_generator==None:
    flimage_generator = tracked_flimage_generate(                                
      matloc=const.TRAINING_TRACKING_MATFILE,                                    
      vidloc=const.TRAINING_VIDEOFILE)                                           
                                                                                 
    convgenerator = convolutions_generate(generator=flimage_generator)           
    evaluation_generator = autoencoder_generator(generator=convgenerator)        
  results = model.evaluate_generator(
    evaluation_generator, 
    val_samples)
  return results

def run(params, trainparametergenerators):
    model = None
    parameters = None
    
    task_envvar_name = 'SLURM_ARRAY_TASK_ID'
    task_id = int(os.environ[task_envvar_name])if task_envvar_name in os.environ else None
    if params['class'] in nnets.paramgen:
      paramgen = trainparametergenerators(params['class'])
      parameters = paramgen(params)
      generator = parameters['model_generator']
      model = generator(parameters['modelparams'])
    else:
      raise InputError("Model {} not recognized".format(params['class']))

    parameters["trainingrun"] = task_id
    parameters['model'] = model
    parameters['intelligent'] = params['intelligent']
    exerciseMachine = ModelTrainer(model, parameters)
    exerciseMachine.collect()
