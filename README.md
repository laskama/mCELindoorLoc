## Multi-Task Neural Network for Position Estimation in Large-Scale Indoor Environments

This repository holds the implementation of the neural network model for multi-task indoor localization (building/floor/position)
estimation in a single forward pass, which was proposed in:

M. Laska and J. Blankenbach, "Multi-Task Neural Network for Position Estimation in Large-Scale Indoor Environments," in IEEE Access, vol. 10, pp. 26024-26032, 2022, doi: [10.1109/ACCESS.2022.3156579](https://ieeexplore.ieee.org/document/9727182). 

The network simultaneously classifies a grid cell and perform within grid cell regression to obtain a final position estimate. This neural network architecture for this is as follows:
![caption](img/mCEL_arch.png)
This allows for obtaining a floor/building classification (by the location of the classified grid cell) as well as a precise position estimate via the regression head via a single forward 
pass of the network. The combination of grid cell classification and within grid cell regression is also more accurate than directly performing regression over the entire floor of a given building as
demonstrated in the paper.

To prevent large errors in case of failures of the grid cell classification head, we introduced the multi-cell encoding learning (m-CEL) technique.
The network is supplied with several redundant encodings by letting the encoding grid slightly overlap. 
<p>
  <img src="img/enc_1.png" width="24%"/>
  <img src="img/enc_2.png" width="24%"/> 
  <img src="img/enc_3.png" width="24%"/>
  <img src="img/enc_4.png" width="24%"/>
</p>
![](img/enc_%5B14%5D_1.png)![](img/enc_%5B14%5D_1.png)
Via m-CEL the network is guided towards also learning the alternative encoding, such that the regression head produces a reasonable estimate in case that the classification head classifies an alternative grid-cell. 
This further improves the accuracy of the final position estimate.

In this repository it is demonstrated how to apply the proposed model on the **giaIndoorLoc** dataset as described in the paper [VI-SLAM2tag](), in which the dataset was introduced together with its generation procedure.

### Installation

Clone the repository and install the required packages as listed in the ```requirements.txt``` file.
Installation via any virtual environment such as *virtualenv* or *conda* are is strongly recommended. 

### Dataset integration

For training the network on the giaIndoorLoc dataset, it has to be downloaded [here](https://www.dropbox.com/sh/gayegcskeyzc5pa/AAB1hSg3aHuBKAJmv5geqfKba?dl=0) and placed in the ```datasets``` folder.

### Train and evaluate network

To train the mCEL network, a ```config.yml``` file has to be set up. Example files that have been used to create the evaluation
described in the [VI-SLAM2tag]() paper are contained in the ```config``` folder. Any parameter not specified in the ```config.yml```
will be taken from the default values given in ```config/default_params.py```.
Afterwards the pipelines specified within the config file can be executed via
```shell
python pipeline.py -c path/to/config.yml
```

### Visualization of predictions
The predictions of the models can be visualized and displayed along with the ground truth position for each fingerprint of the test partition of the dataset.
Just add the flag ```--visualize_predictions``` when executing a pipeline as
```shell
python pipeline.py -c path/to/config.yml --visualize_predictions
```
For each prediction a plotting window such as  
![](img/example.png)
will open. Once the window is closed, the predictions for the next fingerprint of the test dataset will be shown and so on.

### Reproduce results of VI-SLAM2tag paper

In order to reproduce the results from the VI-SLAM2tag paper, the pre-trained model weights have to be downloaded [here](https://www.dropbox.com/sh/m6m0ar69t8pftg6/AAAi18FfTR93pIJGJE8Uq8pga?dl=0), place the 
```exp``` folder within the root of the directory and execute the script ```run_exp.sh```.
You can also set the ```pretrained: False``` in the ```default_params.py``` file and train the model from scratch.
Note that this might result in slightly different results as the training is dependent on the hardware and the tensorflow version.
The pretrained models have been obtained via tensorflow==2.8.0 trained on a Macbook Pro with M1Pro chip (CPU-only).

### Model and dataset extension
The repository has been designed to also provide easy integration with other datasets and models.
You have to provide the following:
- Dataset connector: which extracts the data and stores it in the specified format of the repository (please see source file documentation for details)
- Data provider: transforms the raw data of a dataset connector to the format required by the model
- Model: has to implement setup and evaluation function and might overwrite the fit function.

Please study the documented source code for details.