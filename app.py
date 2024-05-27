# enviornment is Google Colab.

import os
import cv2
import numpy as np
from zipfile import ZipFile
from google.colab import files
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


processed_dir = 'processed_images'
os.makedirs(processed_dir, exist_ok=True)


if (not os.path.exists('/content/extracted_images1/img_align_celeba')):
    with ZipFile('/content/drive/MyDrive/img_align_celeba.zip', 'r') as zip_ref:
        try:
            zip_ref.extractall('/content/extracted_images1')
            zip_ref.close()
        except Exception as e:
            print(f'An error has occured while extracting the zip: {e}')
            zip_ref.close()


image_files = os.listdir('/content/extracted_images1/img_align_celeba')


images = []


def size_images():
    images = []
    for file in image_files:
        input_file_path = f'extracted_images1/img_align_celeba/{file}'
        output_file_path = f'{processed_dir}/{file}'

        image = cv2.imread(input_file_path)

        # Check if the image is loaded correctly
        if image is None:
            print(f"Failed to load {input_file_path}")
            continue

        image = cv2.resize(image, (64, 64))

        image = image / 255.0

        images.append(image)

        cv2.imwrite(output_file_path, image)

    return images


def are_images_processed():
    return len(os.listdir(processed_dir)) > 0


def load_images():
    images = []
    for file in os.listdir(processed_dir):
        file_path = f'{processed_dir}/{file}'

        image = cv2.imread(file_path)

        # Check if the image is loaded correctly
        if image is None:
            print(f"Failed to load {file_path}")
            continue

        images.append(image)

    return images


images = load_images() if are_images_processed() else size_images()
images = np.array(images)


# splitting into sets for training.
X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
              padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
              padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator = make_discriminator_model()
generator = make_generator_model()
BATCH_SIZE = 64


generator.save('generator_model.h5')


discriminator.save('discriminator_model.h5')


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=False)

        generated_images = tf.image.resize(generated_images, [64, 64])

        fake_output = discriminator(generated_images, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


num_steps = 10000

# Training loop

for step in range(num_steps):

    images = X_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

    train_step(images)

    # for every 1000 steps, save the models
    if step % 1000 == 0:
        generator.save('generator_model.h5')
        discriminator.save('discriminator_model.h5')


# loading the models
generator = load_model('generator_model.h5')
discriminator = load_model('discriminator_model.h5')

# use the models.
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# classify the image
decision = discriminator(generated_image)


# Load gen model
generator = load_model('generator_model.h5')

# Load discriminator model
discriminator = load_model('discriminator_model.h5')

# generate an image with the generator:
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)


generator.compile(optimizer='adam', loss=generator_loss)
discriminator.compile(optimizer='adam', loss=discriminator_loss)

# create a random noise vector
noise = tf.random.normal([1, 100])


generated_image = generator(noise, training=False)

# Rescale the image from -1 to 1 to 0 to 1
generated_image = (generated_image + 1) / 2.0

# used to display the image
plt.imshow(generated_image[0])
plt.show()
