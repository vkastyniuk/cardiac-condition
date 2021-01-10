from typing import Tuple, Union, Iterable

import keras


# class InceptionModule(keras.layers.Layer):
#
#     def __init__(self, num_filters: int = 32, kernel_size: int = 40, use_bottleneck: bool = True,
#                  bottleneck_size: int = 32, activation: str = 'linear', **kwargs):
#         super().__init__(**kwargs)
#
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.use_bottleneck = use_bottleneck
#         self.bottleneck_size = bottleneck_size
#         self.activation = activation
#
#     def call(self, inputs: keras.layers.Input, **kwargs):
#         # Step 1
#         z_max_pool = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(inputs)
#
#         input_inception = inputs
#         if self.use_bottleneck and int(inputs.shape[-1]) > 1:
#             input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1, padding='same',
#                                                   strides=1, activation=self.activation, use_bias=False)(inputs)
#
#         # Step 2
#         conv_list = [keras.layers.Conv1D(filters=self.num_filters, kernel_size=1, padding='same', strides=1,
#                                          activation=self.activation, use_bias=False)(z_max_pool)]
#         for kernel_size in (self.kernel_size // (2 ** i) for i in range(3)):
#             conv_list.append(
#                 keras.layers.Conv1D(filters=self.num_filters, kernel_size=kernel_size, padding='same', strides=1,
#                                     activation=self.activation, use_bias=False)(input_inception))
#
#         # Step 3
#         z = keras.layers.Concatenate(axis=2)(conv_list)
#         z = keras.layers.BatchNormalization()(z)
#         z = keras.layers.Activation(activation='relu')(z)
#         return z


num_filters: int = 32
kernel_size: int = 20
use_bottleneck: bool = True
bottleneck_size: int = 32
# activation: str = 'relu'
activation: str = 'linear'


def _inception_module(inputs: keras.layers.Input):
    # Step 1
    z_max_pool = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(inputs)

    input_inception = inputs
    if use_bottleneck and int(inputs.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1, padding='same',
                                              strides=1, activation=activation, use_bias=False)(inputs)

    # Step 2
    conv_list = [keras.layers.Conv1D(filters=num_filters, kernel_size=1, padding='same', strides=1,
                                     activation=activation, use_bias=False)(z_max_pool)]
    for size in (kernel_size // (2 ** i) for i in range(3)):
        conv_list.append(
            keras.layers.Conv1D(filters=num_filters, kernel_size=size, padding='same', strides=1,
                                activation=activation, use_bias=False)(input_inception))

    # Step 3
    z = keras.layers.Concatenate(axis=2)(conv_list)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='relu')(z)
    return z


def shortcut_layer(inputs: Union[keras.layers.Input, Iterable[keras.layers.Input]], z_inception: keras.layers.Input):
    # create shortcut connection
    z_shortcut = keras.layers.Conv1D(filters=int(z_inception.shape[-1]), kernel_size=1, padding='same', use_bias=False)(inputs)
    z_shortcut = keras.layers.BatchNormalization()(z_shortcut)

    # add shortcut to inception
    z = keras.layers.Add()([z_shortcut, z_inception])
    z = keras.layers.Activation(activation='relu')(z)
    return z


def build_model(input_shape: Tuple[int], num_classes: int, num_modules: int = 6, use_residual: bool = True):
    # create series of Inception Module with shortcuts
    input_layer = keras.layers.Input(input_shape)
    z = input_layer
    z_residual = input_layer

    for i in range(num_modules):
        # z = InceptionModule()(z)
        z = _inception_module(z)
        if use_residual and i % 3 == 2:
            z = shortcut_layer(z_residual, z)
            z_residual = z

    gap_layer = keras.layers.GlobalAveragePooling1D()(z)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', keras.metrics.CategoricalAccuracy()])
                  # metrics=[keras.metrics.Accuracy()])
    return model
