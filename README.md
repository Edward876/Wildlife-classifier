

# Animal Classification Model

This project is an animal classification model that identifies various animals from an input image. The model was trained to recognize 90 different animal species, ranging from common pets to wild animals. It is designed to help in classifying animals based on visual features in images.

## Model Overview

The model used in this project is based on the EfficientNet architecture, optimized for image classification tasks. The EfficientNet model was trained on a large dataset of animal images, covering a wide variety of species, to ensure accuracy across a diverse set of animals.

The model is saved in TensorFlow Lite format (`.tflite`) for efficient inference on devices with limited resources. TensorFlow Lite allows the model to be deployed on various platforms, including mobile and embedded systems, without sacrificing performance.

## Project Structure

- `animal_classification_model.tflite`: The pre-trained TensorFlow Lite model used for inference.
- `app.py`: The main script for loading the model, processing input images, and generating predictions.
- `requirements.txt`: Lists the necessary dependencies to run this project.

## Installation

To get started with this project, clone the repository and install the dependencies:

```bash
git clone https://github.com/Edward876/Wildlife-classifier.git
cd animal-classification
pip install -r requirements.txt
```

Ensure that TensorFlow and other required libraries are installed. The specific versions needed are provided in `requirements.txt`.

## Usage

To use the model for animal classification, you can run the main script. Here’s a quick overview of how it works:

1. Load the `animal_classification_model.tflite` model using the TensorFlow Lite Interpreter.
2. Preprocess an input image, resize it to 224x224 pixels, and normalize pixel values.
3. Feed the preprocessed image into the model to get predictions.
4. Map the prediction output to the corresponding animal name.

### Code Example

Here’s a basic example of how to use the model to classify an image:

```python
import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='animal_classification_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess a single image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype('float32')

# Function to predict the animal from an image
def predict_animal(image_path):
    preprocessed_image = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(predictions)
    
    # Animal names corresponding to the model output
    animal_names = [
        'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
        'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog',
        'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
        'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
        'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo',
        'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse',
        'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot',
        'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat',
        'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake',
        'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale',
        'wolf', 'wombat', 'woodpecker', 'zebra'
    ]
    
    return animal_names[predicted_index]

# Example usage
image_path = 'images.jpeg'  # Replace with the path to your image
predicted_animal = predict_animal(image_path)
print("Predicted Animal:", predicted_animal)
```

## Supported Animals

The model can classify 90 animal species, including but not limited to:

- Common pets like cats and dogs.
- Wild animals such as lions, elephants, and zebras.
- Insects like butterflies, beetles, and bees.
- Birds including eagles, parrots, and penguins.

Please see `animal_names` in the code for the full list of supported species.

## Requirements

To run this project, you need the following Python packages:

- `tensorflow`
- `numpy`
- `opencv-python-headless`

These dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Notes

- The model is optimized for CPU-only inference with TensorFlow Lite.
- Ensure input images are clear and contain only a single animal for best results.
- The model’s accuracy may vary depending on lighting, angle, and quality of the input image.

