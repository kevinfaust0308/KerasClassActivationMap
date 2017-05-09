# Keras Class Activation Map Implementation

Python implementation of CAM using keras and tensorflow backend, with both vectorized and non-vectorized versions. Takes in an image and returns a heat map (array format) showing which parts of the image constituted the majority of the classification

## Getting Started

Requires a keras model with a global average pooling layer after the final convolution layer followed by a single input -> output layer.


### Prerequisites

```
matplotlib
keras
numpy
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Original paper: https://arxiv.org/pdf/1512.04150.pdf
