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
        # input_shape = (None, 441000, 1)
        mel_spec = get_melspectrogram_layer(
            input_shape=None,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            window_name="hann_window",
            pad_begin=False,
            pad_end=False,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            mel_f_min=0.0,
            mel_f_max=None,
            mel_htk=False,
            mel_norm="slaney",
            return_decibel=True,
            db_amin=1e-05,
            db_ref_value=1.0,
            db_dynamic_range=80.0,
            input_data_format="default",
            output_data_format="default",
            name="melspectrogram",
        )
        model = Sequential()
        model.add(mel_spec)
        model.add(tf.keras.layers.Permute((2, 1, 3)))
        model.build(input_shape=[None, config.WAVE_LENGTH, 1])
        img_concat = tf.keras.layers.concatenate([model.output] * 3, axis=-1)
        img_norm = tf.keras.layers.BatchNormalization()(img_concat)

        base = tfimm.create_model(config.model_type, pretrained=True, nb_classes=0)
        _, features = base(img_norm, return_features=True)
        gap = tf.keras.layers.GlobalAveragePooling2D(name="GAP")(features["features"])
        gmp = tf.keras.layers.GlobalMaxPooling2D(name="GMP")(features["features"])
        pool_concat = tf.keras.layers.concatenate([gap, gmp], axis=-1)
        drop = tf.keras.layers.Dropout(0.2)(pool_concat)
        logits = tf.keras.layers.Dense(1, name="logits")(drop)

        model = tf.keras.models.Model(inputs=model.input, outputs=logits)
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
