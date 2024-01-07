## Summary of process.py
The process.py script is part of a larger project that involves processing data from the PyTorch MNIST dataset and saving it to a DocArray.

### Function: processAndGetIndex()
This function processes the model and returns an InMemoryExactNNIndex and a DocList.

#### Inputs
None

#### Outputs


### InMemoryExactNNIndex[MNIST]: The indexed document.
DocList[MNIST]: The list of documents.


### Process
The function starts by loading the model using the loadModel() function. The returned model is stored in the doc variable, which is a DocList of MNIST objects.

An instance of InMemoryExactNNIndex for MNIST objects is created and stored in the index variable.

The doc is then indexed using the index object.

Finally, the function returns a tuple containing the index and doc.

This script is part of a larger process that involves getting data from the PyTorch MNIST dataset and saving it to a DocArray.