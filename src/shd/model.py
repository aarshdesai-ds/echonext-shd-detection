"""Model architecture + custom layers.

Kept import-light so the inference image only needs these definitions to
deserialize the trained `.keras` checkpoints. Training-only behaviour (the
augmentation layer) is a no-op at inference time (`training=False`).
"""
from __future__ import annotations

import math
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="shd")
class ECGAugment(layers.Layer):
    """Per-sample ECG augmentation; active only when training=True.

    At inference (`training=False`) this is a pure pass-through, so it adds no
    latency and does not affect predictions.
    """

    def __init__(self, noise_std=0.05, scale_range=0.10, lead_drop_prob=0.10,
                 wander_amp=0.10, max_shift=125, **kw):
        super().__init__(**kw)
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.lead_drop_prob = lead_drop_prob
        self.wander_amp = wander_amp
        self.max_shift = max_shift

    def call(self, x, training=None):
        if not training:
            return x
        import tensorflow as tf
        dt = x.dtype
        b, T, L = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        scale = tf.random.uniform((b, 1, 1), 1.0 - self.scale_range, 1.0 + self.scale_range, dtype=dt)
        x = x * scale
        x = x + tf.random.normal(tf.shape(x), stddev=self.noise_std, dtype=dt)
        t = tf.reshape(tf.linspace(tf.cast(0.0, dt), tf.cast(1.0, dt), T), (1, -1, 1))
        freq = tf.random.uniform((b, 1, 1), 0.5, 2.0, dtype=dt)
        phase = tf.random.uniform((b, 1, 1), 0.0, 2.0 * math.pi, dtype=dt)
        amp = tf.random.uniform((b, 1, 1), 0.0, self.wander_amp, dtype=dt)
        x = x + amp * tf.sin(2.0 * math.pi * freq * t + phase)
        keep = tf.cast(tf.random.uniform((b, 1, L)) > self.lead_drop_prob, dt)
        x = x * keep
        shift = tf.random.uniform([], -self.max_shift, self.max_shift, dtype=tf.int32)
        x = tf.roll(x, shift=shift, axis=1)
        return x

    def get_config(self):
        c = super().get_config()
        c.update(dict(noise_std=self.noise_std, scale_range=self.scale_range,
                      lead_drop_prob=self.lead_drop_prob, wander_amp=self.wander_amp,
                      max_shift=self.max_shift))
        return c


@keras.saving.register_keras_serializable(package="shd")
class AddPositionalEmbedding(layers.Layer):
    """Learnable positional embedding added to a token sequence."""

    def build(self, input_shape):
        L, d = int(input_shape[1]), int(input_shape[2])
        self.pos = self.add_weight(name="pos", shape=(1, L, d),
                                   initializer="random_normal", trainable=True)

    def call(self, x):
        import tensorflow as tf
        return x + tf.cast(self.pos, x.dtype)


@keras.saving.register_keras_serializable(package="shd")
class AttentionPool(layers.Layer):
    """Learned-query attention pooling over the temporal axis -> (B, d)."""

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.proj = layers.Dense(d)
        self.q = self.add_weight(name="query", shape=(d,),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, x):
        import tensorflow as tf
        k = self.proj(x)
        scores = tf.einsum("bld,d->bl", k, tf.cast(self.q, x.dtype))
        w = tf.nn.softmax(scores, axis=1)
        return tf.einsum("bl,bld->bd", w, x)


# The classes needed to deserialize the trained checkpoints.
CUSTOM_OBJECTS = {
    "ECGAugment": ECGAugment,
    "AddPositionalEmbedding": AddPositionalEmbedding,
    "AttentionPool": AttentionPool,
}
