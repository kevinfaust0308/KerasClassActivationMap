# Keras Custom Class Activation Map Implementation (multi CAM overlay options)

Class Activation Map is an unsupervised way of doing object localization with accuracy near par with supervised methods.

CAM is a custom Python implementation of CAM using Keras with a Tensorflow backend. Takes in an image and returns
1. Prediction accuracy (+label and heatmap legend if multi overlay)
2. Heatmap showing which parts of the image constituted the majority of the classification. There is the option to overlay a single or multiple CAM heatmap(s) on top of the original image.

## Getting Started

Within CAM.py, please read docstring of:
1. predict_label_with_cam
2. get_multi_stacked_cam 

Must have a keras model with a global average pooling layer after the final convolution layer followed by a single input -> output layer (Part 1 jupyter notebook)

Part 2 jupyer notebook contains examples of what CAM.py can do

### Prerequisites

```
pip install matplotlib
pip install keras
pip install numpy
```

OPTIONAL: OpenCV for CAM overlay on image

### Results

![Alt text](overlay_img_demo.png?raw=true "Single CAM Overlay")
![Alt text](multi_overlay_img_demo.png?raw=true "Multi CAM Overlay")

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Original paper: https://arxiv.org/pdf/1512.04150.pdf
