import glob, re, shutil, torch
import random, os, json
import numpy as np

import tensorflow as tf
from transformers import AutoTokenizer
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

def dictionize_single_dataset(inputs,tag="train"):
    dict_dataset = {}
    x, y = inputs
    x_ = {}
    x_["source_input_ids"] = x.pop("input_ids")
    x_["source_attention_mask"] = x.pop("attention_mask")
    x_["target_attention_mask"] = x.pop("decoder_attention_mask")

    dict_dataset[f"x_{tag}"] = x_
    dict_dataset[f"y_{tag}"] = {"target_input_ids": y}
    return dict_dataset

def dictionize_t2t_dataset(train_inputs, eval_inputs=None):
    dict_dataset = dictionize_single_dataset(train_inputs, tag="train")
    if eval_inputs is not None:
        dict_dataset.update(dictionize_single_dataset(eval_inputs, tag="eval"))
    return dict_dataset

def get_strategy(args,logger):
    if args.use_tpu:
        # Create distribution strategy
        # checking ip address or tpu name?
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", args.tpu_address):
            args.tpu_address = 'grpc://' + args.tpu_address
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_address)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        logger.info("All TPU devices: ")
        for each_device in tf.config.list_logical_devices('TPU'):
            logger.info(each_device)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        if args.use_gpu:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            logger.info("Number of GPU devices: {}".format(strategy.num_replicas_in_sync))
        else:
            raise ValueError("not available yet")
            # strategy = None
            # logger.info("Using CPU for training")
            # model = model_getter(args)
    return strategy

def create_model(args, logger, model_getter, tokenizer=None, from_pretrained=True, save_args=True):
    # get strategy and Create model
    strategy = get_strategy(args, logger)
    with strategy.scope():
        model = model_getter(args, tokenizer=tokenizer, from_pretrained=from_pretrained)

    logger.info(model.summary())
    # trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # non_trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    # logger.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    # logger.info('Trainable params: {:,}'.format(trainable_count))
    # logger.info('Non-trainable params: {:,}'.format(non_trainable_count))
    # if strategy!=None:
    args.num_replicas_in_sync = strategy.num_replicas_in_sync
    if save_args:
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

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def write_args_enhance(args, logger=None, write_path=None):
    if write_path is None:
        write_path = args.output_path

    with open(os.path.join(write_path, "args.json"), "w+") as f:
        args_dict = {}
        for key, value in args.__dict__.items():
            if is_jsonable(value):
                args_dict[key] = value
        if logger is not None:
            logger.info(json.dumps(args_dict, indent=2))
        else:
            print(json.dumps(args_dict, indent=2))
        f.write(json.dumps(args_dict, indent=2))

def get_existing_cks(output_path, best_ck=False, return_best_ck=False):
    cks_already = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]

    if best_ck:
        for ex in [each for each in cks_already if each.startswith("best")]:
            cks_already.remove(ex)
            shutil.rmtree(os.path.join(output_path, ex))

    index2path = {}

    for each_ck in cks_already:
        if return_best_ck or not each_ck.startswith("best"):
            index2path[int(os.path.basename(each_ck).split("_")[-1])] = os.path.join(output_path, each_ck)

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

def save_and_check_if_early_stop(eval_score, args, logger, model, tokenizer, steps=0, tag="epoch",from_tf=False):
    logger.info("\n")
    logger.info(
        f"*******eval at {tag} = {steps} (gradient accumulation steps={args.__dict__.get('gradient_accumulation_steps', 1)})*********")
    logger.info(f"val_{args.eval_on}: {eval_score}")
    best_save = False
    if args.eval_on == "acc":
        if eval_score >= args.best:
            args.wait = 0
            args.best = eval_score
            logger.info(f"so far the best check point at {tag}={steps} based on eval_on {args.eval_on}")
            save_ck(args, logger, model, tokenizer, steps=steps, tag=tag, best_ck=True,from_tf=from_tf)
            best_save = True
        else:
            args.wait += 1
    else:
        raise ValueError("not support yet")

    logger.info(f"best so far ({args.eval_on}): {args.best}")
    logger.info(f"early stop count: {args.wait}/{args.patience}")
    if not best_save:
        save_ck(args, logger, model, tokenizer, steps=steps, tag=tag, best_ck=False,from_tf=from_tf)

    if args.wait >= args.patience:
        logger.info("run out of patience, early stop")
        return True
    return False

def save_transformer_locally(model_name="bert-base-uncased", save_path=".", is_tf=False):
    """save
    anyone you can find from here: https://huggingface.co/models
    """
    # to use AutoModel, need to install pytorch: pip3 install torch torchvision or pip install torch torchvision
    from transformers import AutoTokenizer, AutoModel, TFAutoModel
    if is_tf:
        model = TFAutoModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    # Load pretrained model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(os.path.join(save_path, model_name))  # save model weights and config
    tokenizer.save_pretrained(os.path.join(save_path, model_name))  # save tokenizer config or/and vocab


def iid_denoise_text(original_text, span_length=3, corrupt_ratio=0.15, lang="zh_cn"):
    """
    This method is implemented for the pre-training objective of T5, as described in the T5 paper (https://arxiv.org/abs/1910.10683)
    this default params setup keeps the same as the original T5 paper on English, we generalize it to more languages such as Chinese
    :param original_text: it is a list of tokens
    :param span_length: 3 for by default as described in T5 paper
    :param corrupt_ratio: 15% by default as described in T5 paper
    :param lang: reserved param for future use
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
    last_replace_pos = - span_length
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


def save_ck(args, logger, model, tokenizer=None, steps=0, tag="epoch", best_ck=False,from_tf=False):
    sorted_indices, index2path = get_existing_cks(args.output_path, best_ck=best_ck)
    if len(sorted_indices) >= args.keep_ck_num:
        logger.info(
            f"there are already {len(sorted_indices)} checkpoints saved that will be more than keep_ck_num={args.keep_ck_num}")
        logger.info(f"hence, remove the oldest one: {index2path[sorted_indices[0]]}")
        shutil.rmtree(
            index2path[sorted_indices[0]])  # remove the oldest checkpoint, i.e., the one with the lowest epoch number
    if best_ck:
        logger.info(
            f'save best model weights and tokenizer to {os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}.h5")}')
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
        else:
            if from_tf:
                model.config.save_pretrained(os.path.join(args.output_path,f"best_ck_at_{tag}_{steps}"))
                model.save_weights(os.path.join(args.output_path,f"best_ck_at_{tag}_{steps}", "tf_model.h5"), overwrite=True)
            else:
                model.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
    else:
        logger.info(
            f'save model weights and tokenizer to {os.path.join(args.output_path, f"ck_at_{tag}_{steps}")}')
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))
        else:
            if from_tf:
                model.config.save_pretrained(os.path.join(args.output_path,f"ck_at_{tag}_{steps}"))
                model.save_weights(os.path.join(args.output_path, f"ck_at_{tag}_{steps}", "tf_model.h5"),
                                   overwrite=True)
            else:
                model.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))


if __name__ == '__main__':
    model_name_or_path = "t5-small"
    from transformers import TFT5ForConditionalGeneration

    model = TFT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    state_dict = load_torch_state_dict_from_h5_weights(model)
    print(len(state_dict))
