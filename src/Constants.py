import math
import os   
import sys
import json

IS_CLUSTER = True
VIDREAD_MODE = ['imageio','cv'][1]
                
PATH_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(
    os.path.join(PATH_PREFIX, "/lib/opencv-2.4.11/lib/python2.7/site-packages")
)
os.system("export LD_LIBRARY_PATH={}/lib/opencv-2.4.11/lib:{}/lib/2.5.0/lib/:$LD_LIBRARY_PATH".format(PATH_PREFIX,PATH_PREFIX))
os.system("export PKG_CONFIG_PATH={}/lib/opencv-2.4.11/lib/pkgconfig/:{}/lib/ffmpeg_built/lib/pkgconfig:$PKG_CONFIG_PATH".format(PATH_PREFIX,PATH_PREFIX))


BACKEND = 'theano'#keras_settings['backend']


WEIGHTS_PATH = os.path.join(PATH_PREFIX, 'weights')
ARCHITECTURE_PATH = os.path.join(PATH_PREFIX, 'architecture') 
MODEL_METRICS_PATH = os.path.join(PATH_PREFIX, 'model_metrics')
TENSORFLOW_LOGS = os.path.join(PATH_PREFIX, 'tensorflow_logs')
MODEL_TRAINING_METADATA = os.path.join(PATH_PREFIX, 'model_metadata.json')

EPOCH_SAMPLES = 500
NEPOCH = 100

TRAININGS_DATA_DIR = os.path.join(PATH_PREFIX, 'trainings')
MAIN_SCRIPT = "/Users/akivalipshitz/Developer/netrain/bin/Main.py"
NORMAL_ENV = "tensorflow"
GPU_ENV = "GPU_tensorflow"
