import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.backend import function


def predict_label_with_cam(model, image_path, last_conv_layer_idx, class_idx=-1, overlay=False, overlay_alpha=0.4):
    ''' (keras.Model, str, int, int, bool, float) -> ndarray, ndarray
    Returns the class activation map of the image with the highest predicted class 
    
    model: a keras model that follows the cam architecture (ConvLayer --> GlobalAveragePoolingLayer -> OutputLayer)
    image_path: path to image 
    last_conv_layer_idx: the index of the final convolutional layer (check with model.summary())
    class_idx: get a cam on class_idx'th class. defaults to highest predicted class
    overlay: get the original image with the cam as an overlay. defaults to false
    overlay_alpha: how transparent to make the cam overlay if overlay is true ([0,1] with 0 being solid color)
    
    Ex: if model predicts image to be a dog, this function returns the dog activation map

    Class activation map is a unsupervised way of doing object localization with accuracy near par
    with supervised methods
    '''

    # read image (height, width, channel)
    original_img = plt.imread(image_path) / 255
    
    # width and height of image
    IMG_HEIGHT, IMG_WIDTH = original_img.shape[0], original_img.shape[1]

    #Reshape into 4d tensor
    img = np.expand_dims(original_img, axis=0)

    #Get the input weights to the final layer 
    class_weights = model.layers[-1].get_weights()[0]

    # variables for easy access to needed layers
    input_layer = model.layers[0]
    final_layer = model.layers[-1]

    # last conv layer (depends on model)
    final_conv_layer = model.layers[last_conv_layer_idx]


    # a function that takes in the input layer and outputs 
    # 1. the feature maps after the last convolutional layer 2. prediction of image
    get_output = function([input_layer.input], [final_conv_layer.output, final_layer.output])

    # call function on our image
    conv_outputs, pred = get_output([img])
    conv_outputs = conv_outputs[0] # 4d tensor to 3d tensor
    
    # #Create the class activation map.
    # we create an empty ?x? array where ? and ? are the dimensions of our feature maps
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])

    # check if we will just return cam of most confident predicted class
    if class_idx == -1:
        # get class index with the highest prediction
        class_idx = np.argmax(pred)

    # get the class weights (WEIGHTSxNUM_CLASSES)
    class_weights = class_weights[:,class_idx]

    ########## VECTORIZED CODE ##########

    # multiply length/width feature map dimensions
    width_times_len = conv_outputs.shape[0]*conv_outputs.shape[1] 
    # our output shape will be the same as our feature map dimensions (w/o # of maps)
    output_shape = conv_outputs.shape[:2] 
    # reshape into 2d
    temp = conv_outputs.reshape((width_times_len,conv_outputs.shape[2])) 
    # multiply all our feature maps with its corresponding weights
    # reshape to class activation map
    cam = np.matmul(temp, class_weights).reshape(output_shape)
    
    #####################################
    
    # . 
    # .
    # .

    ########## NON-VECTORIZED CODE ##########

#     # loop through all our class weights
#     for i in range(len(class_weights)):

#         # get ith class weight and multiply it with ith feature map
#         # matrix-add that to our cam
#         cam += class_weights[i] * conv_outputs[:, :, i]

    #########################################
    
    
    
    # if we want to overlay
    if overlay:
        # resize cam to image dimensions (width, height)
        cam = cv2.resize(cam, (IMG_WIDTH, IMG_HEIGHT)) 
        # change cam to rgb. 'jet' gives classic heat map look
        cmap = plt.get_cmap('jet')
        cam = cmap(cam)
        cam = np.delete(cam, 3, 2)
        
        # make both cam and image have same data types
        original_img = original_img.astype(np.float64)
        cam = cam.astype(np.float64)
        
        # add overlay
        cam = cv2.addWeighted(cam, 1-overlay_alpha, original_img, overlay_alpha, 0)
        
    # return cam and our class predictions
    return cam, pred