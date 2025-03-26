import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

class UNetModel:
    def __init__(self, input_size=(256, 256, 1)):

        self.input_size = input_size
        self.model = self.build_unet()
    
    def conv_block(self, inputs, filters, block_type='down'):

        conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        
        if block_type == 'down':
            pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
            return conv, pool
        return conv
    
    def build_unet(self):

        inputs = layers.Input(self.input_size)
        
        # Contracting Path
        conv1, pool1 = self.conv_block(inputs, 64)
        conv2, pool2 = self.conv_block(pool1, 128)
        conv3, pool3 = self.conv_block(pool2, 256)
        conv4, pool4 = self.conv_block(pool3, 512)
        
        # Bottom
        conv5 = self.conv_block(pool4, 1024, block_type='bottom')
        
        # Expansive Path
        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = layers.concatenate([up6, conv4], axis=-1)
        conv6 = self.conv_block(up6, 512, block_type='up')
        
        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.concatenate([up7, conv3], axis=-1)
        conv7 = self.conv_block(up7, 256, block_type='up')
        
        up8 = layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = layers.concatenate([up8, conv2], axis=-1)
        conv8 = self.conv_block(up8, 128, block_type='up')
        
        up9 = layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = layers.concatenate([up9, conv1], axis=-1)
        conv9 = self.conv_block(up9, 64, block_type='up')
        
        # Final Convolution
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    def train(self, train_generator, validation_generator, epochs=50):

        return self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )
    
    def predict(self, image):

        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return prediction[0]
    
    def visualize_prediction(self, image, mask):

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Segmentation Mask')
        plt.imshow(mask, cmap='gray')
        plt.show()

def prepare_data_generators(image_dir, mask_dir, batch_size=8):

    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rescale=1./255
    )
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed
    )
    
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed
    )
    
    train_generator = zip(image_generator, mask_generator)
    
    return train_generator

def main():
    # Initialize U-Net model
    unet = UNetModel(input_size=(256, 256, 1))
    
    # Prepare data generators
    train_generator = prepare_data_generators(
        image_dir='path/to/train/images', 
        mask_dir='path/to/train/masks'
    )
    
    # Train the model
    history = unet.train(train_generator, epochs=50)
    
    # Optional: Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()