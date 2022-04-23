from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense,
    Softmax,
)
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer
import tensorflow as tf
import tfimm
from config.config import config


def get_model(strategy):

    with strategy.scope():

        base = tfimm.create_model(config.model_type, pretrained=True, nb_classes=0)

        input = tf.keras.Input((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="inp1")
        _, features = base(input, return_features=True)
        gap = tf.keras.layers.GlobalAveragePooling2D(name="GAP")(features["features"])
        gmp = tf.keras.layers.GlobalMaxPooling2D(name="GMP")(features["features"])
        pool_concat = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        drop = tf.keras.layers.Dropout(0.2)(pool_concat)
        logits = tf.keras.layers.Dense(1, name="logits")(drop)

        model = tf.keras.models.Model(inputs=input, outputs=logits)
        model.summary()
        opt = tf.keras.optimizers.Adam(learning_rate=config.LR_START)

        model.compile(
            optimizer=opt,
            # loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
            loss={
                "logits": tf.keras.losses.BinaryCrossentropy(
                    from_logits=True, label_smoothing=0.1
                ),
            },
            # metrics=[
            #    # tf.keras.metrics.SparseCategoricalAccuracy(),
            #    # tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
            #    tf.keras.metrics.CategoricalAccuracy(),
            #    tf.keras.metrics.TopKCategoricalAccuracy(k=5),
            # ],
        )
        return model
