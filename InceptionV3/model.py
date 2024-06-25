# Importing libraries and getting iris datasets
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import pillow_heif
import os

# Register HEIF format with Pillow (for iPhone images)
pillow_heif.register_heif_opener()

# Function to resize and reformat images
def resize_and_reformat_image(input_path, output_path, new_size):
    with Image.open(input_path) as img:
        img = img.resize(new_size, Image.LANCZOS)
        if img.mode == 'RGBA':
            img = img.convert('RGB')    # Convert RGBA to RGB, since Alpha channel is not supported by .JPG files
        img.save(output_path, format='JPEG')

# Process all images in a directory
def process_images_in_directory(input_directory, output_directory, new_size):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp', 'gif', 'heic')):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_directory)
                output_path = os.path.join(output_directory, relative_path)
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_filename = os.path.splitext(filename)[0] + '.jpg'
                output_path = os.path.join(output_dir, output_filename)
                resize_and_reformat_image(input_path, output_path, new_size)

# Image Augmentation
def image_gen_w_aug(train_parent_directory, test_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale = 1/255,
                                       rotation_range = 30,
                                       zoom_range = 0.2,
                                       width_shift_range = 0.1,
                                       height_shift_range = 0.1,
                                       validation_split = 0.15)
    
    test_datagen = ImageDataGenerator(rescale = 1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                        target_size = (224,224),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                        subset = 'training')
    
    validation_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                             target_size = (224,224),
                                                             batch_size = 32,
                                                             class_mode = 'categorical',
                                                             subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                      target_size = (224, 224),
                                                      batch_size = 32,
                                                      class_mode = 'categorical')
    
    return train_generator, validation_generator, test_generator

def model_output_for_TL(pre_trained_model, last_output, num_classes):
    x = Flatten()(last_output)
    
    # Dense hidden layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(pre_trained_model.input, x)
    
    return model

# Define the input and output directories
train_dir = os.path.join('C:/Users/dylan/Documents/AY2024-25/MLAI/Mini Project/MangoOrOrange/Image Datasets/train')
test_dir = os.path.join('C:/Users/dylan/Documents/AY2024-25/MLAI/Mini Project/MangoOrOrange/Image Datasets/test')

# Define the processed directories
processed_train_dir = os.path.join('C:/Users/dylan/Documents/AY2024-25/MLAI/Mini Project/MangoOrOrange/Image Datasets/Processed/train')
processed_test_dir = os.path.join('C:/Users/dylan/Documents/AY2024-25/MLAI/Mini Project/MangoOrOrange/Image Datasets/Processed/test')

# Process images in both directories
process_images_in_directory(train_dir, processed_train_dir, (224, 224))
process_images_in_directory(test_dir, processed_test_dir, (224, 224))

# Generate image data with augmentation
train_generator, validation_generator, test_generator = image_gen_w_aug(processed_train_dir, processed_test_dir)

# Create pre-trained model using InceptionV3 neural network with imagenet weight for image classification
pre_trained_model = InceptionV3(input_shape = (224, 224, 3),
                                include_top = False,
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False  # Freeze the layers, frozen layers won't be updated during training

last_layer = pre_trained_model.get_layer('mixed3')  # Define output as last layer
last_output = last_layer.output

num_classes = 3  # Update the number of classes
model_TL = model_output_for_TL(pre_trained_model, last_output, num_classes)
model_TL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training with adjusted steps per epoch (example calculation)
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history_TL = model_TL.fit(
    train_generator.repeat(),  # Repeat the dataset to avoid running out of data
    steps_per_epoch=steps_per_epoch,  # Adjust this based on your dataset size
    epochs=10,
    verbose=1,
    validation_data=validation_generator.repeat(),  # Repeat the dataset to avoid running out of data
    validation_steps=validation_steps
)

tf.keras.models.save_model(model_TL, 'MangoOrOrange_model.hdf5')
