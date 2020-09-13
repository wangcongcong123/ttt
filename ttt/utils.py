import glob
import random, os, json
import numpy as np

import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, TFAutoModel
import logging

import tensorflow_addons as tfa
from tensorflow import keras
from keras import backend as K

class LRSchudlerCallback(keras.callbacks.Callback):
    def __init__(self, args, logger):
        super(LRSchudlerCallback, self).__init__()
        self.warmup_ratio = args.warmup_ratio
        self.scheduler = args.scheduler
        self.logger = logger

    def on_train_begin(self, logs=None):
        self.steps_per_epoch = self.params["steps"]
        self.epochs = self.params["epochs"]
        self.global_step = 0
        self.logger.info(f"using learning rate scheduler {self.scheduler}")
        if not self.scheduler.startswith("constant"):
            self.total_steps = self.steps_per_epoch * self.epochs
            self.warmup_steps = int(self.total_steps * self.warmup_ratio)
            self.logger.info(
                f"total_steps: {self.total_steps}, steps_per_epoch: {self.steps_per_epoch}, epochs: {self.epochs}, warmup_steps:{self.warmup_steps}")
            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            self.logger.info(f"lr of optimizer to reach through warmup: {K.eval(self.model.optimizer.lr)}")

            self.lr_to_reach = K.eval(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.learning_rate, 0.00)
            self.logger.info(f"now set it to zero for warmup: {K.eval(self.model.optimizer.lr)}")


    def on_train_batch_end(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            if self.scheduler == "warmuplinear" or self.scheduler == "warmupconstant":
                inc = self.lr_to_reach / self.warmup_steps
                K.set_value(self.model.optimizer.learning_rate, K.eval(self.model.optimizer.lr) + inc)
        else:
            if self.scheduler == "warmuplinear" or self.scheduler == "constantlinear":
                dec = self.lr_to_reach / (self.total_steps - self.warmup_steps)
                K.set_value(self.model.optimizer.learning_rate, K.eval(self.model.optimizer.lr) - dec)

        self.global_step += 1

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f"at epoch={epoch}, the learning_rate is {K.eval(self.model.optimizer.lr)}")

    def on_train_end(self, logs=None):
        self.logger.info("testing")


def get_callbacks(args, inputs, logger, eval_getter):
    tqdm_callback = tfa.callbacks.TQDMProgressBar(metrics_format="{name}: {value:0.8f}",
                                                  epoch_bar_format="{n_fmt}/{total_fmt}{bar} ETA: {remaining}s - {desc}, {rate_fmt}{postfix}", )
    lr_scheduler_callback = LRSchudlerCallback(args, logger)
    if args.do_eval == True:
        eval_callback = eval_getter(inputs["x_eval"], inputs["y_eval"], args)
        # return [tqdm_callback,eval_callback]
        return [tqdm_callback, eval_callback, lr_scheduler_callback]
    else:
        return [tqdm_callback, lr_scheduler_callback]


def create_model(args, logger, model_getter,tokenizer=None):
    if args.use_tpu:
        # Create distribution strategy
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + args.tpu_address)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        logger.info("All TPU devices: ")
        for each_device in tf.config.list_logical_devices('TPU'):
            logger.info(each_device)
        strategy = tf.distribute.TPUStrategy(tpu)
        # Create model
        with strategy.scope():
            model = model_getter(args,tokenizer=tokenizer)
    else:
        if args.use_gpu:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            logger.info("Number of GPU devices: {}".format(strategy.num_replicas_in_sync))
            # Open a strategy scope.
            with strategy.scope():
                model = model_getter(args,tokenizer=tokenizer)
        else:
            raise ValueError("not available yet")
            # strategy = None
            # logger.info("Using CPU for training")
            # model = model_getter(args)
    # logger.info(model.summary())
    # trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # non_trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    # logger.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    # logger.info('Trainable params: {:,}'.format(trainable_count))
    # logger.info('Non-trainable params: {:,}'.format(non_trainable_count))
    # if strategy!=None:
    args.num_replicas_in_sync = strategy.num_replicas_in_sync
    write_args(args.output_path, args)
    return model, strategy

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_select)
    tokenizer.save_pretrained(args.output_path)
    return tokenizer

def add_filehandler_for_logger(output_path, logger, out_name="train"):
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    fileHandler = logging.FileHandler(os.path.join(output_path, f"{out_name}.log"), mode="a")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

def set_seed(seed):
    tf.random.set_seed(
        seed
    )
    random.seed(seed)
    np.random.seed(seed)

def write_args(output_path, args):
    with open(os.path.join(output_path, "args.json"), "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

def get_existing_cks(output_path, best_ck=False):
    cks_path_already = glob.glob(os.path.join(output_path, "ck*.h5"))

    if best_ck:
        for ex in glob.glob(os.path.join(output_path, "best*.h5")):
            os.remove(ex)

    index2path = {int(os.path.basename(each_ck_path).split(".")[0].split("_")[-1]): each_ck_path for
                  each_ck_path in cks_path_already}
    sorted_indices = sorted(index2path)  # index here refers to the epoch number
    return sorted_indices, index2path

def load_torch_state_dict_from_h5_weights(model):
    import torch
    state_dict = {}
    for layer in model.layers:
        for resource_variable in layer.weights:
            key = resource_variable.name
            value = torch.tensor(resource_variable.numpy())
            state_dict[key] = value
    total_params_num = sum([element.numel() for element in state_dict.values()])
    print(f"the number of params: {total_params_num}")
    return state_dict

def save_transformer_locally(model_name="bert-base-uncased",save_path=".",is_tf=False):
    """save
    anyone you can find from here: https://huggingface.co/models
    """
    if is_tf:
        model = TFAutoModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    # Load pretrained model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path,exist_ok=True)
    model.save_pretrained(os.path.join(save_path,model_name))  # save model weights and config
    tokenizer.save_pretrained(os.path.join(save_path,model_name))  # save tokenizer config or/and vocab

def iid_denoise_text(original_text, span_length=4, corrupt_ratio=0.10, lang="zh_cn"):
    """
    This method is implemented for the pre-training objective of T5, as described in the T5 paper (https://arxiv.org/abs/1910.10683)
    this default params setup follows the original T5 paper on English, we transfer it to Chinese here
    :param original_text: default to chinese text, it is list of words split by space for english
    :param span_length: 4 for chinese (intuitively makes more sense), and 3 for english as in T5 paper
    :param corrupt_ratio: 15% for english and 10% for chinese since we use single word as the corrpution tagret
    intuitively, 10% in Chinese lower than 15% as in English could be more appropriate
    :return:
    """
    source_text = []
    target_text = []
    # if lang == "en":
    #     corrupt_ratio = 0.15
    #     span_length = 3  # 3 as in T5 paper
    # make deterministic for reproducibility
    # random.seed(2020)
    replace_i = 0
    skip_count = span_length
    last_replace_pos = -span_length
    for pos in range(len(original_text)):
        if skip_count < span_length - 1:
            skip_count += 1
        else:
            if random.uniform(0, 1) < corrupt_ratio:
                extra_token = f"<extra_id_{replace_i}>"
                if pos != last_replace_pos + span_length:
                    target_text.append(extra_token)
                to_replace_span = original_text[pos: pos + span_length]
                target_text.extend(to_replace_span)
                source_text.append(extra_token)
                replace_i += 1
                skip_count = 0
                last_replace_pos = pos
            else:
                source_text.append(original_text[pos])
    if target_text == "" or target_text == []:
        target_text.append("<extra_id_0>")
    return original_text, source_text, target_text


if __name__ == '__main__':
    model_name_or_path = "t5-small"
    from transformers import TFT5ForConditionalGeneration
    model = TFT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    state_dict = load_torch_state_dict_from_h5_weights(model)
    print(len(state_dict))