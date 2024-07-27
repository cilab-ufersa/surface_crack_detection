# Surface Crack Detection

## About 

This project is a deep learning model to detect cracks on civil engineering building elements. The model is based on the U-Net architecture and SAM (Segment Anything Model) loss function. The dataset used to train the model is the [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) dataset. 

## Getting Started

To run the project, you need to follow the steps below:

### Installation

```bash
    $ git clone
    $ cd surface_crack_detection
```

### Prerequisites

What things you need to have to be able to run:

  * Python 3.11 +
  * Pip 3+
  * VirtualEnvWrapper is recommended but not mandatory


### Requirements 

```bash
    $ pip install -r requirements.txt
```

### Running the project

Segmentation of the image:

```bash
    $ python unet_resnet50.py
```



![Figure](/dataset/result.png)

##  Publications related to this project

Bezerra, P. H. A., H. C. Dantas, L. M. G. Morais, and R. C. B. Rego. ["A Deep Learning Artificial Intelligence Algorithm to Detect Cracks on Civil Engineering Building Elements."](https://github.com/cilab-ufersa/surface_crack_detection/blob/develop/surface_crack_detection/CINPAR2024.pdf) In: XX International Conference on Building Pathology and Constructions Repair, 2024, Fortaleza. *XX International Conference on Building Pathology and Constructions Repair*. Fortaleza/CE, 2024. v. 1.
