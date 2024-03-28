import numpy as np
import tensorflow as tf

def conv2D_block(input_tensor, num_filters):
  x = tf.keras.layers.Conv2D(
      filters=num_filters,
      kernel_size=(3,3),
      padding="same",
      kernel_initializer="he_normal"
  )(input_tensor)

  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation("relu")(x)

  x = tf.keras.layers.Conv2D(
      filters=num_filters,
      kernel_size=(3,3),
      padding="same",
      kernel_initializer="he_normal"
  )(x)

  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation("relu")(x)

  return x

def encoder_block(input_img, num_filters, dropout):
  x = conv2D_block(input_img, num_filters)
  p = tf.keras.layers.MaxPooling2D((2, 2))(x)
  p = tf.keras.layers.Dropout(dropout)(p)

  return x, p

def decoder_block(inputs, skip, num_filters):
  x = tf.keras.layers.Conv2DTranspose(
      filters=num_filters,
      kernel_size=(3, 3),
      strides=(2, 2),
      padding="same"
  )(inputs)

  x = tf.keras.layers.Concatenate()([x, skip])
  x = conv2D_block(x, num_filters)

  return x

def unet(input_shape, num_filters=16, dropout=0.25):
  input_image = tf.keras.layers.Input(input_shape)

  # encoder block
  s1, p1 = encoder_block(input_image, num_filters*1, dropout)
  s2, p2 = encoder_block(p1, num_filters*2, dropout)
  s3, p3 = encoder_block(p2, num_filters*4, dropout)
  s4, p4 = encoder_block(p3, num_filters*8, dropout)

  # bridge
  b1 = conv2D_block(p4, num_filters*16)

  # decoder block
  d1 = decoder_block(b1, s4, num_filters*8)
  d2 = decoder_block(d1, s3, num_filters*4)
  d3 = decoder_block(d2, s2, num_filters*2)
  d4 = decoder_block(d3, s1, num_filters*1)

  output = tf.keras.layers.Conv2D(
      1,
      kernel_size=(1, 1),
      activation="sigmoid"
  )(d4)

  model = tf.keras.models.Model(inputs=input_image, outputs=output)

  return model