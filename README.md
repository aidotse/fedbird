# fedbird

## Setup 

1.Copy the Annotation.zip folder in the data directory and unzip it.  
2.Build the docker image using  
> sh build.sh  


## Starting the training in Federated mode  

> sh run_serevr.sh  
> sh run_client1.sh  
> sh run_client2.sh  



### Remaining tasks

1. Write map benchmarking code to check the final map values  
2. Experiments and benchmarking  


# Use of Federated Learning Framework

fl_interface directory contain the tools to use differnt framework
of federated learning: FEDn, HPe, PySyft.

## FEDn

Each client will generate its own container. Scripting is on going.  A
dockerfile is provided by platform e.g. nvidia_platform.dockerfile,
and ensure that FEDn SDK and FedBird code are deployed in the container.
The Docker compose will start the client.

command line to run a client:
1. cd fl_interface/FEDn
2. bash start.sh dns test nvidia_platform.dockerfile

Note: not clear year how to make deployement easy yet.... suggestion
welcome.