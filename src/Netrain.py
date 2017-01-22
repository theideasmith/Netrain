import sys
import os
sys.path.append('/jukebox/murthy/akiva/tracker/src')
import modeltrain
from Constants import *
import subprocess
import AutoEncoderArchitectures as aea
import argparse
import json
from utilities import ifnone
os.system('source {}'.format(os.path.join(PATH_PREFIX, 'jobsinit.sh')))

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument(
    '--architecture',
    help="architecture to use",
    default='simple_dense',
    type=str)

parser.add_argument(                                                              
    '--class',                                                                    
    help="Which class of model to use", 
    default="autoencoder")                                                        

parser.add_argument(
    '--archpath',
    help='specify a directory where model architectures should be saved to',
    default=ARCHITECTURE_PATH,
    type=str)

parser.add_argument(
    '--wpath',
    help='specifies directory where model weights should be stored',
    default=WEIGHTS_PATH,
    type=str)

parser.add_argument(
  '--metricspath',
  help='where to store model evaluation metrics',
  default=MODEL_METRICS_PATH,
  type=str)

parser.add_argument(
    '--nb_epoch',
    help='specify number of epochs to train for',
    default = MCARLO_NB_EPOCH,
    type=int)
parser.add_argument(
    '--epoch_samples',
    help='number of samples per epoch',
    default=MCARLO_EPOCH_SAMPLES,
    type=int) 
parser.add_argument(
    '--nnets',
    help='number of networks to train',
    default=1,
    type=int)
parser.add_argument(
    '--intelligent',
    help="Stops training once a loss stops going down",
    default=False,
    type=bool)
parser.add_argument(
    '--paramscheme',
    help="Which parameter generation scheme to use",
    default="default",
    type=str)

def parseargs(argstring):
  args = parser.parse_args(argstring)   
  arguments = vars(args)                
  return arguments

def run(argstring, paramgens):
  parsedargs = parseargs(argstring)
  print parsedargs
  print "BEGINNING TRAIN LOOP"
  for i in xrange(parsedargs['nnets']):
    print "Network {}/{}".format(i,parsedargs['nnets'] )
    modeltrain.run(parsedargs, paramgens)

class Netrain:
  def __init__(self, modelparamgenerator):
    self.genparams = modelparamgenerator

  def run(argv = sys.argv[1:]):
    run(argv, self.genparams)
