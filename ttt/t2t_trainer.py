''''
this is a customize trainer for T5-like mode training,
in this class, the training loop is customized for more flexibility and control over
'''
import math
import os
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from keras import backend as K
from ttt.utils import add_filehandler_for_logger, get_existing_cks
from tensorboardX import SummaryWriter
# for translation evaluation from: https://github.com/mjpost/sacrebleu
# which is also used in the original T5 paper
import sacrebleu
from .utils import write_args_enhance

class T2TTrainer():
    def __init__(self, args, logger):
        self.eval_on = args.eval_on
        assert self.eval_on in ["acc",
                                "bleu"], "now t2t training only supports --eval_on acc, bleu, only works when --do_eval=True"
        # self.best = -np.Inf

        self.patience = args.patience
        self.wait = 0
        self.logger = logger
        self.args = args
        self.use_tb = self.args.__dict__.get('use_tb', False)

        self._tb_writer = None
        if self.use_tb:
            self._tb_writer = SummaryWriter(log_dir=self.args.__dict__.get('output_folder', "runs"))
        self.scheduler = args.scheduler

        if "learning_rate" in self.args.__dict__:
            self.lr_to_reach = args.learning_rate
        else:
            self.lr_to_reach = args.lr

        self.args.best = np.Inf if self.args.eval_on == "loss" or self.args.eval_on == "perplexity" else - np.Inf
        self.best = self.args.best

    def save_ck(self, model, steps, tag="epoch", best_ck=False):
        sorted_indices, index2path = get_existing_cks(self.args.output_path, best_ck=best_ck)

        if len(sorted_indices) >= self.args.keep_ck_num:
            self.logger.info(
                f"there are already {len(sorted_indices)} checkpoints saved that will be more than keep_ck_num={self.args.keep_ck_num}")
            self.logger.info(f"hence, remove the oldest one: {index2path[sorted_indices[0]]}")
            os.remove(index2path[sorted_indices[
                0]])  # remove the oldest checkpoint, i.e., the one with the lowest epoch number

        if best_ck:
            self.logger.info(
                f'save best model weights to {os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5")}')
            model.save_weights(os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5"),
                               overwrite=True)
        else:
            self.logger.info(
                f'save model weights to {os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5")}')
            model.save_weights(os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5"),
                               overwrite=True)

    def train(self, model, strategy, tokenizer, inputs, evaluate_fn=None,verbose=False):
        x_train, y_train = inputs["x_train"], inputs["y_train"]
        num_train_examples = len(inputs["y_train"]["target_input_ids"])
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        if self.args.do_eval:
            assert "x_eval" in inputs and "y_eval" in inputs, "do_eval=True, and no validation data is found"
            x_val, y_val = inputs["x_eval"], inputs["y_eval"]
            eval_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            eval_dataset = eval_dataset.batch(self.args.eval_batch_size)
            val_length = math.ceil(
                len(inputs["y_eval"]["target_input_ids"]) / (self.args.eval_batch_size))

        global_batch_size = self.args.per_device_train_batch_size * strategy.num_replicas_in_sync
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(global_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

        # THERE WILL BE exceptions when switching to distributed_dataset when running on tpus if
        # val_dist_dataset = strategy.experimental_distribute_dataset(eval_dataset)
        train_length = math.ceil(num_train_examples / global_batch_size)
        self.steps_per_epoch = train_length

        if verbose:
            self.logger.info(model.summary())

        # these are used for non-constant lr scheduler
        if "num_train_epochs" in self.args.__dict__:
            self.args.num_epochs_train = self.args.num_train_epochs
        if "log_and_save_steps" in self.args.__dict__:
            self.args.log_steps = self.args.log_and_save_steps

        self.total_steps = self.steps_per_epoch * self.args.num_epochs_train

        if "warmup_steps_or_ratio" in self.args.__dict__:
            if self.args.warmup_steps_or_ratio <= 1 and self.args.warmup_steps_or_ratio > 0:
                self.args.warmup_steps = int(self.total_steps * self.args.warmup_steps_or_ratio)
            else:
                self.args.warmup_steps = self.args.warmup_steps_or_ratio
        else:
            self.args.warmup_steps = int(self.total_steps * self.args.warmup_ratio)

        self.warmup_steps = self.args.warmup_steps
        write_args_enhance(self.args, logger=self.logger)

        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(lr=self.args.lr if self.scheduler.startswith("constant") else 0.0)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )

            def compute_loss(labels, predictions):
                per_example_loss = loss_fn(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

            def train_step(x_train, y_train):
                with tf.GradientTape() as tape:
                    # here some changes has been made (compared to before commit `a07c58e` ) to fix a bug reported here: https://github.com/wangcongcong123/ttt/issues/2
                    # The following describes how this bug is fixed
                    # the compute_loss function in transformers:TFT5ForConditionalGeneration has already taken care of the loss computation (already averaged!!!!) that failed
                    # when switching to TPU, hence we re-compute it here using the returned logits from the model ready for backprop instead of using the internally calculated loss
                    outputs = model(inputs=x_train["source_input_ids"], attention_mask=x_train["source_attention_mask"],
                                    decoder_attention_mask=x_train["target_attention_mask"],
                                    labels=y_train["target_input_ids"], training=True, return_dict=True)
                    logits = outputs.logits
                    loss = compute_loss(tf.reshape(y_train["target_input_ids"], (-1, y_train["target_input_ids"].shape[-1])),
                                        tf.reshape(logits, (-1, logits.shape[-1])))

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            @tf.function
            def distributed_train_step(x_train, y_train):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(x_train, y_train,))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            # evaluate
            def evaluate(steps, tag="epoch"):
                assert tag in ["epoch", "global_step"]
                gts = []
                preds = []
                for x_eval, y_eval in tqdm(eval_dataset, total=val_length, desc="evaluating..."):
                    predictions = model.generate(input_ids=x_eval["source_input_ids"],
                                                 attention_mask=x_eval["source_attention_mask"],
                                                 max_length=self.args.max_tgt_length)
                    pred = [tokenizer.decode(ids) for ids in predictions]
                    gt = [tokenizer.decode(ids) for ids in y_eval["target_input_ids"]]
                    # labels (not -100 replaced since it is not used to calculate loss here)
                    preds.extend(pred)
                    gts.extend(gt)

                if self.eval_on == "bleu":
                    # bleu = 0
                    bleu = sacrebleu.corpus_bleu(preds, [gts])
                    eval_score = bleu.score
                else:
                    eval_score = accuracy_score(gts, preds)
                    self.logger.info(f"val_cls_report: {classification_report(gts, preds, digits=4)}")

                if self.use_tb:
                    self._tb_writer.add_scalar(f"val_{self.eval_on}_{tag}", eval_score, steps)

                self.logger.info("\n")
                self.logger.info(f"*******eval at {tag} = {steps} on validation dataset*********")
                self.logger.info(f"val_{self.eval_on}: {eval_score}")

                if self.eval_on == "acc" or self.eval_on == "bleu":
                    if eval_score >= self.best:
                        self.wait = 0
                        self.best = eval_score
                        self.logger.info(
                            f"so far the best check point at {tag}={steps} based on eval_on {self.eval_on}")
                        self.save_ck(model, steps, tag, best_ck=True)
                    else:
                        self.wait += 1
                else:
                    raise ValueError("not support yet")

                self.logger.info(f"best so far({self.eval_on}): {self.best}")
                self.logger.info(f"early stop count: {self.wait}/{self.patience}")
                self.save_ck(model, steps, tag)
                if self.wait >= self.patience:
                    self.logger.info("run out of patience, early stop")
                    if self.use_tb:
                        self._tb_writer.close()
                    sys.exit(0)

            def update_lr(global_step):
                # already tested on tpu, works fine
                # global_step is dynamically passed here
                if global_step <= self.warmup_steps:
                    if self.scheduler == "warmuplinear" or self.scheduler == "warmupcostant":
                        inc = self.lr_to_reach / self.warmup_steps
                        K.set_value(optimizer.learning_rate, K.eval(optimizer.lr) + inc)
                else:
                    if self.scheduler == "warmuplinear" or self.scheduler == "constantlinear":
                        dec = self.lr_to_reach / (self.total_steps - self.warmup_steps)
                        K.set_value(optimizer.learning_rate, K.eval(optimizer.lr) - dec)
                # for "constant" scheduler, nothing to do here

            global_step = 0
            early_exit = False
            interval_loss = 0.0
            interval_count = 0
            for epoch in tqdm(range(self.args.num_epochs_train), desc="epochs"):

                self.logger.info(f"start training at epoch = {epoch}")
                self.logger.info(f"global train batch size = {global_batch_size}")
                self.logger.info(f"using learning rate scheduler: {self.scheduler}")
                self.logger.info(
                    f"num_train_examples: {num_train_examples}, total_steps: {self.total_steps}, steps_per_epoch: {self.steps_per_epoch}")
                if self.scheduler != "constant":
                    self.logger.info(f"warmup_steps:{self.warmup_steps}")


                pbar = tqdm(enumerate(train_dist_dataset), total=train_length)
                for step, (x_train, y_train) in pbar:
                    # learning rate scheduler
                    update_lr(global_step)
                    loss = distributed_train_step(x_train, y_train)
                    interval_loss += loss.numpy()
                    interval_count += 1
                    global_step += 1
                    pbar.set_description(f"training - epoch {epoch + 1}/{self.args.num_epochs_train} iter {step}: train loss {loss.numpy():.5f}. lr {optimizer.lr.numpy():e}")

                    if self.args.log_steps != -1 and global_step % self.args.log_steps == 0:
                        if self.use_tb:
                            self._tb_writer.add_scalar("train_loss_global_step", interval_loss / interval_count,
                                                       global_step)
                            self._tb_writer.add_scalar("train_lr_global_step", optimizer.lr.numpy(), global_step)

                        if self.args.do_eval:
                            if evaluate_fn is not None:
                                eval_dict = evaluate_fn(self.args, self.logger, model, tokenizer, eval_dataset, steps=global_step, tag="global_step")
                                if self._tb_writer:
                                    if "eval_scores" in eval_dict:
                                        for key, value in eval_dict["eval_scores"].items():
                                            self._tb_writer.add_scalar(f"eval_{key}_global_step", value, global_step)
                                if "is_early_stop" in eval_dict and eval_dict["is_early_stop"]:
                                    self.logger.info(f"run out of patience at global step = {global_step}, early stop")
                                    if self._tb_writer:
                                        self._tb_writer.close()
                                    early_exit = True
                                    break
                            else:
                                evaluate(global_step, tag="global_step")
                        self.logger.info(f"train loss at global_step {global_step}: {interval_loss / interval_count}")
                        interval_loss = 0.0
                        interval_count = 0
                if early_exit:
                    break

                train_loss = interval_loss / interval_count
                interval_loss = 0.0
                interval_count = 0
                if self.args.log_steps == -1:
                    if self.args.do_eval:
                        if evaluate_fn is not None:
                            eval_dict = evaluate_fn(self.args, self.logger, model, tokenizer, eval_dataset, steps=epoch + 1, tag="epoch")
                            if self._tb_writer:
                                if "eval_scores" in eval_dict:
                                    for key, value in eval_dict["eval_scores"].items():
                                        self._tb_writer.add_scalar(f"eval_{key}_epoch", value, epoch + 1)
                            if "is_early_stop" in eval_dict and eval_dict["is_early_stop"]:
                                self.logger.info(f"run out of patience at epoch = {epoch + 1}, early stop")
                                if self._tb_writer:
                                    self._tb_writer.close()
                                break
                        else:
                            evaluate(epoch, tag="epoch")
                    if self.use_tb:
                        self._tb_writer.add_scalar("train_loss_epoch", train_loss,
                                                   global_step)
                        self._tb_writer.add_scalar("train_lr_epoch", optimizer.lr.numpy(), global_step)
                    self.logger.info(f"train loss at end of epoch {epoch}: {train_loss}")

                if not self.args.do_eval:
                    # if do not do evaluate, the checkpoint at the end of epoch needs to be saved
                    self.save_ck(model, epoch + 1, tag="epoch")

            if self.use_tb:
                self._tb_writer.close()
