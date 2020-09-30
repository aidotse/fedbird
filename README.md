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

Script files allow to start the component on host.
List of command:
- bash start_data_storage.sh (launch minio, mongodb and mongo-express)
- bash start_dashboard.sh --component-path components/dashboard/ (launch the FEDn dashboard)
- bash start_reducer.sh --component-path components/reducer/ (launch the reducer)
- bash start_combiner.sh --platform nvidia --component-path components/combiner/ --config config/combiner.yaml --port 12080 --certificate certificates/reducer-cert.pem (launch a combiner - need to implement the name of the combiner like it is done in client)
- bash start_client.sh --platform nvidia --config config/client.yaml --component-path components/client --client-name bird_nest_1 --certificate certificates/reducer-cert.pem
