from DocArray_mnist.process import processAndGetIndex 
from icecream import ic
from docarray.index import InMemoryExactNNIndex
from DocArray_mnist.Model import MNIST, oneHotEncode
from docarray import DocList
from typing import Tuple, List
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np

data: Tuple[InMemoryExactNNIndex[MNIST], DocList[MNIST]] = processAndGetIndex()

def getSimilar(limit: int = 10, query: int = 0) -> Tuple[List[MNIST], List[float]]:
    results, scores = data[0].find(query= oneHotEncode(query), limit=limit, search_field="label")
    return results.tensor, scores

def getImages(limit: int = 10, query: int = 0):
    results, scores = getSimilar(limit=limit, query=query)
    results = [np.array(image.reshape(28, 28)) for image in results]
    return results

# plt.figure(figsize=(20, 4))
# for index, (image, score) in enumerate(zip(results.tensor, scores)):
#     plt.subplot(2, 5, index + 1)
#     plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
#     plt.title('Score: %.2f' % score, fontsize = 20)

# plt.show()

def main():
    # gradio
    with gr.Blocks() as demo:
        
        number = gr.Number(value=0)
        limit = gr.Number(value=10)

        gallary = gr.Gallery(
            label="Results",
            object_fit="contain",
            height="300px",
        )

        btn = gr.Button(value="Search")
        btn.click(fn=getImages, inputs=[limit, number], outputs=gallary)

    demo.launch()

if __name__ == "__main__":
    main()
