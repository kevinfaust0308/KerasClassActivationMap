# Keras Class Activation Map Implementation

Class Activation Map is an unsupervised way of doing object localization with accuracy near par with supervised methods.

CAM.py is a Python implementation of CAM using keras with a tensorflow backend. Takes in an image and returns 1. predicted label probability and 2. a heat map showing which parts of the image constituted the majority of the classification. Overlay option available to see the CAM on the image

## Getting Started

Please read docstring of CAM.py. Must have a keras model with a global average pooling layer after the final convolution layer followed by a single input -> output layer.

Documented 2-part jupyter notebook tutorial included


### Prerequisites

```
pip install matplotlib
pip install keras
pip install numpy
```

OPTIONAL: OpenCV for CAM overlay on image

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Original paper: https://arxiv.org/pdf/1512.04150.pdf
