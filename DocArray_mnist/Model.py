from docarray import BaseDoc, DocList
from docarray.typing import ImageTensor, NdArray
from icecream import ic
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np


class MNIST(BaseDoc):
    tensor: ImageTensor = None
    label: NdArray = None

transform = transforms.Compose([
    transforms.ToTensor(),
])


def transformToTensor(data):
    return transform(data)

def oneHotEncode(target):
    return np.eye(10)[target]

def process(data, target):
        doc = MNIST()
        doc.tensor = transformToTensor(data)
        doc.label = oneHotEncode(target)
        return doc

def loadModel() -> DocList[MNIST]:
    ic("Loading from pytorch")
    dataset = datasets.MNIST('./data', train=True, download=True)
    docs = DocList[MNIST]()


    for i in tqdm(range(len(dataset))):
        data, target = dataset[i]
        docs.append(process(data, target))

    ic("Length of docs: ", len(docs.tensor))
    return docs
