
# Deep Learning for Stokes Profile Synthesis
**Author**: Leonardo Barberi

## Description
This repository hosts the implementation of the algorithms to achieve an automatization of Stokes Profile Synthesis using deep learning.
It includes processing scripts for pre-processing the data, training the models, and evaluating the results.

## Learning Problem
The learning problem is formulated as a regression task, where the input is a set of physical parameters that uniquely determine Stokes profiles, and the output is 
the corresponding Stokes profile. Due to the high-dimensional nature of Stokes profiles, we employ a deep learning model
to learn the mapping between the input parameters and a compact representation of the output profiles, using functional principal component analysis.

## Functional Principal Component Analysis
Analogously to multivariate principal component analysis (PCA), functional PCA is an analysis technique that recovers the 
principal components, i.e. the functions that best capture the variability of a set of functions.

### Visualizations
The figures below illustrate various aspects of the structures analyzed in the study.

#### Stokes Profiles
<img src="images/all_profiles.png" width="400">
<figcaption> All Stokes I Profile functions present in the dataset. Each color represents one function. </figcaption>

#### Functional Principal Component Analysis
<img src="images/2PCs.png" width="400">
<figcaption> The first two functional principal components pertaining to the Stokes I profiles in our dataset </figcaption> 

#### Functional Principal Component Analysis
<img src="images/PC_reconstruction.png" width="400">
<figcaption> Reconstruction of a sample Stokes I profile function using 2 vs 14 principal components </figcaption> 


## Installation
To reproduce the analysis environment, you will need Python 3.9 or later. Please install the required Python packages listed in `requirements.txt`.

```bash
git clone git@github.com:lbarberi1927/solar_spectropolarimetry.git
cd solar_spectropolarimetry
pip install -r requirements.txt
```

To have the environment on SIRIUS, I recommend to set up an SFTP connection and transfer the repository to the cluster.

## Data
The data employed in our analysis is sourced from simulations ran by IRSOL - Istituto Ricerche Solari Locarno.
To train the model, create the files and directories as specified in the `configs/data` file. This is very important to 
specify both the directories from which to read the data and to save the preprocessed data for training.

To prepare the data for training, run the following scripts. These are idealised to run locally or on a high-performance
computing cluster. However, not on SIRIUS, as the computing capabilities are not sufficient to run the training.

Travel to the source directory of the repository in the SIRIUS cluster and execute the `single_file_transfer.py`:

```bash
cd solar_spectropolarimetry
python3 -m src.preprocess.single_file_transfer
```

You must then download the preprocessed data to your local machine, or to a computing cluster, for example using the `scp` command:

```bash
scp barberi@sirius:solar_spectropolarimetry/data.nosync/fe6302_basic.csv barbele@rosa.usi.ch:solar_spectropolarimetry/data.nosync/
```

It is important that the file is saved in the same relative directory as in SIRIUS. The file name is specified in the `configs/data.py` file.
Next, run the following script locally or on the computing cluster to finish preprocessing the data:

```bash
python3 -m src.preprocess.preprocess
```

Your data folder will now be populated with a directory for each stokes profile. Each directory will contain the
labels for the deep learning model, while the input variables are saved in the parent folder, since they are shared for 
all profiles. Both input and response variables are already split for training and testing.

## Training
To train the deep learning model, a sbatch script is provided to run the training on a high-performance computing cluster.
Run the following commands from the source directory to train the model with CUDA:

```bash
module load python
module load cuda
sbatch runner.sh
```

To track the progress of the training, you can see the status of the job using `squeue` and the logs of the training
using `cat output.txt`. For example, you can run the following to give you periodic updates. To stop printing updates use Ctrl+C:
```bash
while true; do squeue; cat output.txt; sleep 60; done
```

Training configurations can be modified in the `configs/NN_config.py` file. The computing resources needed for training 
are specified in the `runner.sh` script.

If you wish to run training locally, simply run the following command:
```bash
python3 -m src.core.train
```

The trained model will be saved in the `saved_models` directory, as specified in `configs/NN_config.py`.

## Deployment
To deploy the trained model, run the following command:

```bash
python3 -m src.core.deploy
```
Then simply follow the instructions in the terminal to input parameters with which to call the models.
To exit the prediction loop, type `exit`.

