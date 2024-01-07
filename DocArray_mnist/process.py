# get data from pytorch mnist and save it to DocArray

from typing import Tuple
from icecream import ic
from DocArray_mnist.Model import MNIST, loadModel
from docarray import DocList
from docarray.index import InMemoryExactNNIndex

def processAndGetIndex() -> Tuple[InMemoryExactNNIndex[MNIST], DocList[MNIST]]:
    """
    Process the model and return the InMemoryExactNNIndex and the DocList.

    Returns:
        Tuple[InMemoryExactNNIndex[MNIST], DocList[MNIST]]: The indexed document and the DocList.
    """
    doc: DocList[MNIST] = loadModel()
    index = InMemoryExactNNIndex[MNIST]()

    ic("Indexing")
    index.index(doc)

    return index, doc

