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

        input = tf.keras.Input((config.IMG_FREQ, config.IMG_TIME, 3), name="inp1")
        _, features = base(input, return_features=True)
        ## SED model flow
        # (batch_size, freq, frames, channels, )
        
        # (batch_size, frames, channels, )
        freq_reduced = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x,axis=1), output_shape=None)(features["features"])
        ## without pooling
        #ap = tf.keras.layers.AveragePooling1D(pool_size=2,strides=1,)(freq_reduced)
        
        dd = tf.keras.layers.Dense(2048)(freq_reduced)
        ## Starting with the attention block
        att = tf.keras.layers.Conv1D(2,1,strides=1,padding='valid',)(dd)
        att_aten = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(tf.keras.activations.tanh(x)), output_shape=None)(att)
        cla = tf.keras.layers.Conv1D(2,1,strides=1,padding='valid',)(dd)
        cla_aten = tf.keras.layers.Lambda(lambda x: tf.keras.activations.sigmoid(x), output_shape=None)(cla)
        
        x = att_aten * cla_aten
        x = tf.keras.layers.Lambda(lambda x:tf.math.reduce_sum(x, axis=1), output_shape=None, name='logits')(x)
        
        #model = tf.keras.models.Model(inputs=input, outputs=[x, att_aten, cla_aten])
        model = tf.keras.models.Model(inputs=input, outputs=x)
        #model = tf.keras.models.Model(inputs=input, outputs=logits)
        model.summary()
        opt = tf.keras.optimizers.Adam(learning_rate=config.LR_START)

        model.compile(
            optimizer=opt,
            # loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
            loss={
                "logits": tf.keras.losses.CategoricalCrossentropy(
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
