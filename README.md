# Fedbird

Note: This repo consists of only the client code. To setup the  reducer, combiner and mongodb  as shown in the image below, clone this [repo](https://github.com/scaleoutsystems/fedn/tree/v0.2.3) and follow the instructions in it's readme. The code has been tested with V0.2.3 tag of fedn
 

## Introduction 

Fedbird is a poc developed as  part of the federated learning project funded by Vinnova. The POC involves training an object detection model in a federated learning setting using the baltic seabird dataset. More details about the dataset and experiment setup are given below.


### Dataset
The dataset consists of 2000 hours of video footage of the guillemots on a ledge on Stora Karlsö in Gotland. Researchers from SLU and AI Sweden have manually annotated 1800 frames taken from these video materials.

### Annotations
The annotations provide the bounding boxes of the birds present on the ledge classifying them into Adult birds, Chicks and Eggs.
Quality of the dataset: real-world, challenging, class-imbalanced dataset.

### Experimental setup
The current dataset consists of images from two camera view points(two different CCTV cameras) pertaining to two different ledges on the Island. Each CCTV camera is allotted one Edge device for processing the data stream coming from it. The Edge device is also responsible for training a local object detection model using the bounding box annotations for some selected images from the data stream. The local models from both the clients are sent to the central server for aggregation.

### Object detection Model 

For this POC we have used YOLOv3−tiny as our baseline model for object detection. The choice was dictated by considering conditions such as  the architecture of the model. The architecture  supports relatively small available dataset, small number of classes and facilitates fast training speeds on an edge device like Xavier.

### Framework 

We use an open source framework called Fedn to communicate between the clients and the central server. In the Fedn high level model, there are three principal layers such as clients, combiners and reducers. The example is also useful for general scalability tests in fully distributed mode. 

## Network Topology

- 2 Edge nodes as clients
- 3 nodes - reducer , combiner and database
- Communication - Fedn by Scaleout
- Tensorflow/keras


![Alt text](https://github.com/aidotse/fedbird/blob/master/images/unnamed.png)
 
## Setting up a client

### Provide local training and test data
This code repo assumes that trainig and test data annotations are available at
 
- fedbird/data/clientn/train.txt
- fedbird/data/clientn/Annotation_images 
- fedbird/data/clientn/val.txt


Get in touch with Ebba(ebba.josefson@ai.se) to get access to the dataset. But simple changes to the code can make it work for any object detection dataset.

### Creating a compute package
To train a model in FEDn you provide the client code (in 'client') as a tarball (you set the name of the package in 'settings-reducer.yaml'). For convenience, we ship a pre-made package. Whenever you make updates to the client code (such as altering any of the settings in the above mentioned file), you need to re-package the code (as a .tar.gz archive) and copy the updated package to 'packages'. From 'test/mnist':

```bash
tar -cf fedbird.tar client
gzip fedbird.tar
cp fedbird.tar.gz packages/
```

## Creating a seed model
The baseline CNN is specified in the file 'client/init_model.py'. This script creates an untrained neural network and serialized that to a file, which is uploaded as the seed model for federated training. For convenience we ship a pregenerated seed model in the 'seed/' directory. If you wish to alter the base model, edit 'init_model.py' and regenerate the seed file:

```bash
sh run_seed.sh  
```

## Start the client

### If you are testing a psuedo distributed system : 

The easiest way to start clients for quick testing is by using Docker. We provide a docker-compose template for convenience. First, edit 'fedn-network.yaml' to provide information about the reducer endpoint. Then:

```bash
sudo docker-compose -f docker-compose.dev.yaml up --scale client=2 
```
> Note that this assumes that a FEDn network is running (see separate deployment instructions). The file 'docker-compose.dev.yaml' is for testing againts a local pseudo-distributed FEDn network. Use 'farralon3.yaml' and 'roaster3.yaml' if you are connecting against a reducer part of a distributed setup and provide a 'extra-hosts-farallon3.yaml' or 'extra-hosts-roaster3.yaml' respectively for each of the clients

> Use the .xavier files if you want to test it out on the xaviers in a distributed mode.

> Use build_on_xavier to build the client docker images on the xavier. For the roaster3 client, Open the build_on_xavier in a text editor and edit the extra-hosts-farallon3.yaml file to extra-hosts-roaster3.yaml , farallon3.yaml to roaster3.yaml. 

> Change the docker image name to 'fedn-roaster3-clent' in run_on_xavie.sh file while running on the roaster3 client. 
