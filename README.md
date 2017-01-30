## Netrain

Netrain aims to simplify the process of training a huge number of keras ANNs on the PNI clusters. It was originally developed as a bookkeeping system for use when training thousands of networks for hyperparameter optimization. Though it is now clear that training thousands of networks is computationally impractical given, Netrain remains a valuable tool to have for any project which involves experimenting with a large number of ANN models. 

Netrain tool enables you to run highly configurable ANN training schedules and keep track of them in a very organized fashion. It generates a self contained project for each batch of networks you train and helps you keep track of all your batches. This keeps your networks organized and makes it very easy to annotate certain training runs, make changes to the code for particular networks, or retrain certain networks. with Netrain everything stays organized and modular. 

## Workflow

Netrain answers the call of duty as soon as you have a bunch of network models you want to train. Netrain allows you to organize your models by type and specify which models of which type to train. Say you are working on $3$ autoencoder models and $4$ classifier models, the former group named `"auto"` and the latter group named `"class"`,  Netrain allows you to specify exactly which models from these two groups to train. In netrain language, a model's group is called its `"class"` and its specific structure is called its `"architecture"`. 



The general workflow is as follows:

1. **Configuration**
   - Configure training and evaluation data generator functions for each network model
   - Configure training specifications for each model
   - Configure a model generator function; this allows you to alter different hyperparameters for the same general model structure. 
2. **Build**
   - Specify (a) a group of networks to train and (b) parameters for training (number of epochs, batches per epoch, etc)
   - Use the netrain build tool to generate a self contained project for this training run.  
   - Submit an array job to the cluster. 

What you'll eventually want to do is to generate a self contained project for that batch and store it in some  first step is to generate a self contained project for this batch of networks. To set this up you will need to follow these steps: 

## 1 – Specifications

**Tell Netrain Where Training Files and Data is Stored**

Netrain builds a new mini project each time you train a set of networks. First, you will need to specify the directory to which Netrain will write each of these self contained training codebases. Create such a directory and one you have done that, find the `Constants.py` file in `src`. Search for the constant `TRAININGS_DATA_DIR`. It should say `os.path.join(PATH_PREFIX, 'trainings')` .Replace this with the directory you created to which netrain will write training code.  

**Tell Netrain how to load up regular tensorflow or GPU tensorflow**

In order for training to work properly, you must install a regular tensorflow and a gpu tensorflow into a conda environment. Then, open `Constants.py` and set `GPU_ENV` to the name of the conda environment with gpu tensorflow, and `NORMAL_ENV` to the name of conda environment with a regular tensorflow. 

NOTE: if you are not on a GPU enabled machine then you can just use the non-gpu conda environment for both

## 2 – Tell netrain about your networks

The only way netrain can train networks is if it knows which networks to train. 

- First, you'll need to get the Netrain API into your python path so you can use it in your code. 
- In your project directory, make a file called `netrain_main.py`. Every time a job starts running on the cluster, this file `netrain_main.py` is responsible for implementing the training of the models. `netrain_main.py` is passed command line arguments which specify the parameters for the model training delegated to it. 
- Navigate to the netrain  `src`  folder in Netrain and open the `Constants.py` file. Find the variable named `MAIN_SCRIPT` and set it to `path/to/netrain_main.py`. 
- Then, go back and insert the following into `netrain_main.py` 

```python
import os
os.path.append("path/to/netrain/src/")
from Netrain import Netrain
```

- Now that you have imported `Netrain` into this script, you'll need to instantiate a `Netrain` application object and pass it a **model configuration function**. Read on to see what this function does. 

## 3 – Model Configuration Function

The model configuration function is integral. It is passed to the `Netrain` application instance and returns a dictionary containing all the information necessary for Netrain to automatically get models running. 

As an prior step, append the following to your `netrain_main.py`:

````python
app = Netrain(model_configuration_function)
````

Now, you'll need to actually define `model_configurator`. Remember that `netrain_main.py` is passed command line arguments to specify which models it should train? The `model_configuration_function` is called by the `Netrain` application instance. It is passed a dictionary of input parameters and must return a dictionary of output parameters, both of which are specified below.  The paramters it returns are the ones ultimately used for training. 

### Configuration Function Input

| Field                               | Value Specification                      |
| ----------------------------------- | :--------------------------------------- |
| `"class"`                           | A `string` specifying  group of the model to be trained by this job. In our case, this would be either `"auto"` or `"class"` |
| `"architecture"`                    | The name of the model within the group `Class`. |
| `"nb_epoch"`                        | The number of epochs all models trained by this script should run for. |
| `"epoch_samples"`                   | How many samples per epoch for all models trained by this script |
| `"nnets"                          ` | The number of models trained by this script. NOTE: they will all have the same architecture. The option to train multiple models in one script is useful if (1) the cluster is busy so you want to take advantage of allocated jobs by training multiple models with them, (2) you want to do hyperparameter optimization for this model of these models. |
| `"intelligent"`                     | Whether the model training should auto-terminate once a certain error threshold is met |
| `"wpath"`                           | Path to the directory to which mdoel weights are stored |
| `"archpath"`                        | Path to the directory to which architectures are stored |
| `"metricspath"`                     | Path to the directory to which metrics are stored |
| `"trainingrun"`                     | When training using sbatch, this input variable specifies the number array job of the current process. So if you had wanted to train 100 modelclass.modelarch models each on its own process, then this variable will let you know which process is the current process; perhaps if you wanted to make modifications to each of the models for hyperparameter optimization. |
|                                     |                                          |

Once passed a dictionary with this data, the `model_configuration_function` must return a dictionary with the following data all present. If all fields are not present, then something will break. So you must be sure to be positive all fields are present. 

### Configuration Function Output

| `"name"`                 |                                          |
| ------------------------ | :--------------------------------------- |
| `"nb_epoch"`             | The number of epochs to train this model with |
| `"archpath"`             | The directory where architectures are stored (so you can actually use them after training). You can just use the `archpath` from the configuration function's input dictionary. |
| `"wpath"`                | The directory where model weights are stored (obviously the most integral directory of them all). You can just use the `wpath` from the configuration function's input dictionary. |
| `"metricspath"`          | The directory where model metrics are stored. You can just use the `model` from the configuration function's input dictionary. |
| `"model_generator"`      |                                          |
| `"training_generator"`   | A generator function that returns training data for the model. Output ought to be a two-tuple of the form `(x, y)` where `x` is the measured input and `y` is the measured output which the model tries to predict. |
| `"evaluation_generator"` | A generator function that returns evaluation data for the model. Output ought to be a two-tuple of the form `(x, y)` where `x` is the measured input and `y` is the measured output which the model tries to predict. |
| `"optimizer"`            | Which optimization scheme to use (see keras optimizers https://keras.io/optimizers/ for the valid options) |
| `"loss"`                 | Which loss function to use for the model (see keras objectives for the valid options https://keras.io/objectives/) |

Anytime you want to introduce a new model you can add the necessary logic to your configuration function

### WARNING – Make sure `wpath` and `archpath` have enough space to store all your networks and their architectures before you begin training.

## 4 – Finishing Setup

Finally, append the following to `netrain_main.py`

```python
app.run()
```

So that in toto it reads:

```python
import os
os.path.append("path/to/netrain/src/")
from Netrain import Netrain
from mycode import model_configuration_function
app = Netrain(model_configuration_function)
app.run()
```

## 5 – Running 

Now's the fun part when everything comes together. Navigate to the `build` directory where you should find a script called `trainscripts_builder`. It has three commands, `generate`, `add` and `show`. 

- **Generate** builds a training project directory for a specified group of models
- **add** adds new networks in a training project
- **show** gives an overview of the status of all training projects  

For example, let's say you want to train three models, the first two of type `classifer`, named `clu_one` and `chransynth`, and the third of type `autoencoder`, named `classical`. You would do the following

`./trainscripts generate --models="classifier.clu_one=10, chransynth=32, autoencoder.classical=45 --gpu --array=200 --desc="Initial training test" --epoch_samples=10000 --nb_epoch=1000`

Then, navigate to the directory created which will be printed as output after running the above command. You can either submit all the models at once: 

1. To submit all models at once, run `bash runbatch.sh`
2. To submit only particular models, navigate to `scripts/` and for each model you want to train, enter `bash modelclass-modelarch-sbatch.sh`. 

This will submit your models to be trained on the cluster to your specification. 



```
usage: trainscripts_builder generate [-h] [--gpu] [--models MODELS]
                                     [--array ARRAY] [--nb_epoch NB_EPOCH]
                                     [--epoch_samples EPOCH_SAMPLES]
                                     [--run RUN] [--intelligent INTELLIGENT]
                                     [--desc DESC]

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 whether to use GPU. Only turn this on if you are using
                        aGPU enabled cluster. NOTE: to use GPU you must first
                        load in tensorflow.Do this by installing Tensorflow
                        into a conda environment. See the tensorflow section
                        in the Netrain guide for more information
  --models MODELS       A comma separated list of the format modelclass.modelt
                        ype=number_of_times_to_train_this_model_for_each_job
                        specifying which models to run
  --array ARRAY         Number of array jobs to run. For every array job, all
                        the models are trained. Depending on what you specify
                        in the "models" option, each time a job is run it can
                        train an arbitrary number of the SAME model
  --nb_epoch NB_EPOCH   number of keras epochs to train models for
  --epoch_samples EPOCH_SAMPLES
                        How many samples the model trains per epoch
  --run RUN             Whether the batches should be sent to spock
                        immediately after running this script
  --intelligent INTELLIGENT
                        < True | False > Whether to stop training model when
                        error stop changing
  --desc DESC           Optionally add a description for this batch of models.
                        This is useful if you want to keep track of which
                        models you trained when
```

