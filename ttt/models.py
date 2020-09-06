
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel, TFT5ForConditionalGeneration
import os
from transformers.modeling_tf_utils import get_initializer


def get_lr_metric(optimizer):
    def lr(x, y):
        return optimizer.lr

    return lr

def create_sl_cls_model(model_name_or_path, input_seq_length, args):
    ## transformer encoder
    encoder = TFAutoModel.from_pretrained(model_name_or_path)

    encoder_config = encoder.config
    if not os.path.isfile(os.path.join(args.output_path, "config.json")):
        encoder_config.save_pretrained(args.output_path)

    input_ids = layers.Input(shape=(input_seq_length,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(input_seq_length,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(input_seq_length,), dtype=tf.int32)

    if "distilbert" in args.model_select:
        # distilbert does not allow to pass token_type_ids
        sequence_outs = encoder(input_ids, attention_mask=attention_mask)[0]
    else:
        sequence_outs = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

    # according to modeling_tf_bert:TFBertPooler. In transformers, models like ROBERTA and Electra do not ooffer direct outputs of pooled_output
    # to make it genelisable, the pooler is re-written here
    # this may not have a big effect on perf. if simply replacing the following pooler with this: pooled_output=sequence_outs[:, 0]
    pooled_output = tf.keras.layers.Dense(
        encoder_config.hidden_size,
        kernel_initializer=get_initializer(encoder_config.initializer_range),
        activation="tanh",
        name="dense",
    )(sequence_outs[:, 0])


    if hasattr(encoder_config,"hidden_dropout_prob"):
        pooled_output = tf.keras.layers.Dropout(encoder_config.hidden_dropout_prob)(pooled_output, training=True)
    else:
        pooled_output = tf.keras.layers.Dropout(encoder_config.dropout)(pooled_output, training=True)

    logits = tf.keras.layers.Dense(len(args.label2id), name="classifier", use_bias=False)(pooled_output)
    probs = layers.Activation(keras.activations.softmax)(logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=probs,
    )

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=args.lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', get_lr_metric(optimizer)])
    return model


def create_t2t_model(model_name_or_path, args):
    ## transformer encoder
    encoder = TFT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    encoder_config = encoder.config
    if not os.path.isfile(os.path.join(args.output_path, "config.json")):
        encoder_config.save_pretrained(args.output_path)
    return encoder


def get_model(args):
    if args.task == "single-label-cls":
        return create_sl_cls_model(args.model_select, args.input_seq_length, args)
    elif args.task == "t2t" or args.task=="translation":
        return create_t2t_model(args.model_select, args)
    else:
        # when more task are supported -> todo
        pass