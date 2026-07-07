"""Neural network architectures used in the manuscript experiments."""

from __future__ import annotations


def build_dcnn_sa(
    input_features: int,
    learning_rate: float = 5e-4,
    attention_heads: int = 4,
    weight_decay: float = 1e-3,
    use_attention: bool = True,
):
    """Build the two-convolution, attention, three-dense DCNN model.

    The input is treated as a one-dimensional feature sequence.  The
    self-attention block replaces a conventional pooling layer, matching the
    architecture described in the manuscript.
    """

    import tensorflow as tf

    regularizer = tf.keras.regularizers.l2(weight_decay)
    inputs = tf.keras.Input(shape=(input_features,), name="selected_features")
    x = tf.keras.layers.Reshape((input_features, 1), name="feature_sequence")(inputs)

    x = tf.keras.layers.Conv1D(
        32,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_regularizer=regularizer,
        name="conv_1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)

    x = tf.keras.layers.Conv1D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_regularizer=regularizer,
        name="conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)

    if use_attention:
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=max(8, 64 // attention_heads),
            name="self_attention",
        )(x, x)
        x = tf.keras.layers.Add(name="attention_residual")([x, attn])
        x = tf.keras.layers.LayerNormalization(name="attention_norm")(x)
    else:
        x = tf.keras.layers.MaxPooling1D(pool_size=2, name="max_pool")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer, name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer, name="fc_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DCNN_SA" if use_attention else "DCNN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auroc", curve="ROC"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )
    return model
