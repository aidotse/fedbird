# FedBird project

1. Create a folder "data" in the project root folder and unpack the data Annotation-seabird.zip inside it.
2. Create the package file from the root folder:
~~~~
tar -czvf fedbird.tar.gz client
~~~~
3. Stand inside the seed folder call the init_model: 
~~~~
python init_model.py
~~~~
This creates a seed model located and named "seed/birdcage"
(If this was in the fedn repo you could follow the instructions from the test examples. Add the model "birdcage" in the reducer ui)
