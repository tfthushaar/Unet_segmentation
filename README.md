U-Net for Image Segmentation

Overview:
--------------------------------------------------
This project contains a comprehensive implementation of U-Net for image segmentation, designed for binary segmentation tasks. It leverages TensorFlow and Keras to build, train, and evaluate the U-Net model on your dataset.

--------------------------------------------------
1. Overview:
--------------------------------------------------
The U-Net architecture is widely used for biomedical image segmentation. This implementation provides a fully configurable U-Net model, data generators with augmentation for robust training, visualization tools for predictions, and guidance on recommended datasets and preprocessing steps.

--------------------------------------------------
2. Model Architecture:
--------------------------------------------------
The core implementation is encapsulated in the UNetModel class.
- Contracting Path: Uses convolutional blocks that include two Conv2D layers followed by Batch Normalization and a MaxPooling layer for downsampling.
- Bottleneck: The deepest layer employs similar convolutional blocks.
- Expansive Path: Upsamples feature maps, concatenates with corresponding features from the contracting path, and applies convolutional blocks.
- Final Layer: A 1x1 convolution with sigmoid activation outputs the segmentation mask.

--------------------------------------------------
3. Installation:
--------------------------------------------------
Ensure you have Python 3.x installed along with the necessary libraries. Install the dependencies using the following command:

pip install numpy tensorflow matplotlib

Note: This implementation uses TensorFlow's Keras API. Ensure that your TensorFlow version is compatible with your hardware and desired performance optimizations.

--------------------------------------------------
4. Usage:
--------------------------------------------------
To run the implementation:

a. Clone the repository or copy the code into your working directory.
b. Prepare your dataset by organizing your images and corresponding masks in separate directories. Make sure that images and masks are resized to a consistent size (e.g., 256x256) and normalized.
c. Configure the data generators by updating the paths in the prepare_data_generators function to point to your training images and masks directories.
d. Train the Model: Run the main() function to start training. The script will plot the training and validation loss after training completes.

Command:
python unet_segmentation.py

e. Visualize Predictions: After training, use the predict and visualize_prediction functions in the UNetModel class to see how the model performs on new images.

--------------------------------------------------
5. Data Preparation:
--------------------------------------------------
The script uses the ImageDataGenerator from Keras to perform data augmentation on-the-fly. Augmentation parameters include:
- Rotation range
- Width and height shifts
- Horizontal flipping
- Rescaling of images

Example augmentation configuration:
rotation_range: 20 degrees,
width_shift_range: 0.1,
height_shift_range: 0.1,
horizontal_flip: True,
rescale: 1/255

--------------------------------------------------
6. Training:
--------------------------------------------------
The model is compiled with the following settings:
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Metrics: Accuracy

The train method of UNetModel handles the training loop and returns a history object for plotting training progress.

--------------------------------------------------
7. Datasets & Resources:
--------------------------------------------------
Recommended Datasets for Biomedical Segmentation include:

a. Medical Segmentation Decathlon:
   - A diverse set of medical images for various organ segmentation tasks.
   - Website: http://medicaldecathlon.com/

b. ISIC Skin Lesion Dataset:
   - High-quality annotated images for skin cancer lesion segmentation.
   - Website: https://challenge.isic-archive.com/

c. Lung Nodule Analysis Dataset (LUNA16):
   - CT scan images for lung nodule segmentation.
   - Website: https://luna16.grand-challenge.org/

--------------------------------------------------
8. References:
--------------------------------------------------
- U-Net: Convolutional Networks for Biomedical Image Segmentation. (https://arxiv.org/abs/1505.04597)
- TensorFlow Keras Documentation. (https://www.tensorflow.org/api_docs/python/tf/keras)
- Keras ImageDataGenerator. (https://keras.io/api/preprocessing/image/)


