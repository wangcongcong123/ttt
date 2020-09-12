''''
this is a customize trainer for T5-like mode training,
in this class, the training loop is customized for more flexibility and control over
'''
import logging
import math
import os
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from keras import backend as K

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from ttt.utils import add_filehandler_for_logger, get_existing_cks

from tensorboardX import SummaryWriter
# for translation evaluation from: https://github.com/mjpost/sacrebleu
# which is also used in the original T5 paper
import sacrebleu


class T2TTrainer():
    def __init__(self, args):
        self.eval_on = args.eval_on
        assert self.eval_on in ["acc", "bleu"], "now t2t training only supports --eval_on acc, bleu, only works when --do_eval=True"

        self.best = -np.Inf

        # score = bleu_metric.compute(preds, gts)
        self.patience = args.patience
        self.wait = 0
        self.args = args
        self.use_tb = self.args.use_tb
        if self.use_tb:
            self._tb_writer = SummaryWriter(log_dir=os.path.join("runs", args.output_folder))

        self.scheduler = args.scheduler
        self.lr_to_reach = args.lr
        self.warmup_ratio = args.warmup_ratio
        add_filehandler_for_logger(args.output_path, logger)

    def save_ck(self, model, steps, tag="epoch", best_ck=False):
        sorted_indices, index2path = get_existing_cks(self.args.output_path, best_ck=best_ck)

        if len(sorted_indices) >= self.args.keep_ck_num:
            logger.info(
                f"there are already {len(sorted_indices)} checkpoints saved that will be more than keep_ck_num={self.args.keep_ck_num}")
            logger.info(f"hence, remove the oldest one: {index2path[sorted_indices[0]]}")
            os.remove(index2path[sorted_indices[
                0]])  # remove the oldest checkpoint, i.e., the one with the lowest epoch number
        # write_args(self.args.output_path, self.args)

        if best_ck:
            logger.info(
                f'save best model weights to {os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5")}')
            model.save_weights(os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5"),
                               overwrite=True)
        else:
            logger.info(
                f'save model weights to {os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5")}')
            model.save_weights(os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5"),
                               overwrite=True)

    def train(self, model, strategy, tokenizer, inputs):

        x_train, y_train = inputs["x_train"], inputs["y_train"]
        num_train_examples=len(y_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((*x_train, y_train))
        if self.args.do_eval:
            assert "x_eval" in inputs and "y_eval" in inputs, "do_eval=True, and no validation data is found"
            x_val, y_val = inputs["x_eval"], inputs["y_eval"]
            val_dataset = tf.data.Dataset.from_tensor_slices((*x_val, y_val))
            val_dataset = val_dataset.batch(self.args.eval_batch_size * strategy.num_replicas_in_sync)
            val_length = math.ceil(len(y_val) / (self.args.eval_batch_size * strategy.num_replicas_in_sync))

        global_batch_size = self.args.per_device_train_batch_size * strategy.num_replicas_in_sync
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(global_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

        # THERE WILL BE exceptions when switching to distributed_dataset when running on tpus if
        # val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
        train_length = math.ceil(len(y_train) / global_batch_size)
        self.steps_per_epoch = train_length
        # these are used for non-constant lr scheduler
        self.total_steps = self.steps_per_epoch * self.args.num_epochs_train
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)


        with strategy.scope():
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

            # learning rate starts from zero if it is constant - non-decay ones (i.e., warmuplinear, warmupconstant, and warmupcosine
            optimizer = tf.keras.optimizers.Adam(lr=self.args.lr if self.scheduler.startswith("constant") else 0.0)

            def train_step(inputs):
                with tf.GradientTape() as tape:
                    # inputs[0]: input_ids
                    # inputs[1]: attention_mask
                    # inputs[2]: shifted(right) decoder input ids
                    # inputs[3]: decoder_attention_mask
                    # inputs[4]: lm labels (not shifted)
                    logits = \
                        model(inputs=inputs[0], attention_mask=inputs[1], decoder_input_ids=inputs[2],
                              decoder_attention_mask=inputs[3], training=True)[0]
                    loss = compute_loss(tf.reshape(inputs[4], (-1, inputs[4].shape[-1])),
                                        tf.reshape(logits, (-1, logits.shape[-1])))

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss

            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            # evaluate
            def evaluate(steps, tag="epoch"):
                assert tag in ["epoch", "global_step"]
                gts = []
                preds = []

                for inputs in tqdm(val_dataset, total=val_length, desc="evaluating..."):
                    predictions = model.generate(input_ids=inputs[0],
                                                 attention_mask=inputs[1],
                                                 max_length=self.args.max_tgt_length)
                    pred = [tokenizer.decode(ids) for ids in predictions]
                    gt = [tokenizer.decode(ids) for ids in inputs[-1]]

                    preds.extend(pred)
                    gts.extend(gt)

                if self.eval_on == "bleu":
                    # bleu = 0
                    bleu = sacrebleu.corpus_bleu(preds, [gts])
                    eval_score=bleu.score
                else:
                    eval_score = accuracy_score(gts, preds)
                    logger.info(f"val_cls_report: {classification_report(gts, preds, digits=4)}")

                if self.use_tb:
                    self._tb_writer.add_scalar(f"val_{self.eval_on}_{tag}", eval_score, steps)

                logger.info("\n")
                logger.info(f"*******eval at {tag} = {steps} on validation dataset*********")
                logger.info(f"val_{self.eval_on}: {eval_score}")

                if self.eval_on == "acc" or self.eval_on == "bleu":
                    if eval_score >= self.best:
                        self.wait = 0
                        self.best = eval_score
                        logger.info(f"so far the best check point at {tag}={steps} based on eval_on {self.eval_on}")
                        self.save_ck(model, steps, tag, best_ck=True)
                    else:
                        self.wait += 1
                else:
                    raise ValueError("not support yet")

                logger.info(f"best so far({self.eval_on}): {self.best}")
                logger.info(f"early stop count: {self.wait}/{self.patience}")
                self.save_ck(model, steps, tag)
                if self.wait >= self.patience:
                    logger.info("run out of patience, early stop")
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
            for epoch in tqdm(range(self.args.num_epochs_train), desc="epochs"):

                logger.info(f"start training at epoch = {epoch}")
                logger.info(f"global train batch size = {global_batch_size}")
                logger.info(f"using learning rate scheduler: {self.scheduler}")
                logger.info(f"num_train_examples: {num_train_examples}, total_steps: {self.total_steps}, steps_per_epoch: {self.steps_per_epoch}")
                if self.scheduler != "constant":
                    logger.info(f"warmup_steps:{self.warmup_steps}")

                epoch_total_loss = 0.0
                num_batches = 0
                pbar = tqdm(enumerate(train_dist_dataset), total=train_length)
                for step, x in pbar:
                    # learning rate scheduler
                    update_lr(global_step)

                    loss = distributed_train_step(x)
                    epoch_total_loss += loss.numpy()
                    global_step += 1
                    num_batches += 1
                    pbar.set_description(
                        f"training - epoch {epoch + 1}/{self.args.num_epochs_train} iter {step}: train loss {loss.numpy():.5f}. lr {optimizer.lr.numpy():e}")

                    if self.args.log_steps != -1 and global_step % self.args.log_steps == 0:
                        if self.use_tb:
                            self._tb_writer.add_scalar("train_loss_global_step", epoch_total_loss / num_batches,
                                                       global_step)
                            self._tb_writer.add_scalar("train_lr_global_step", optimizer.lr.numpy(), global_step)

                        if self.args.do_eval:
                            evaluate(global_step, tag="global_step")
                        logger.info(f"train loss at global_step {global_step}: {epoch_total_loss / num_batches}")

                train_loss = epoch_total_loss / num_batches

                if self.args.log_steps == -1:
                    if self.args.do_eval:
                        evaluate(epoch, tag="epoch")
                    if self.use_tb:
                        self._tb_writer.add_scalar("train_loss_epoch", epoch_total_loss / num_batches,
                                                   global_step)
                        self._tb_writer.add_scalar("train_lr_epoch", optimizer.lr.numpy(), global_step)
                    logger.info(f"train loss at end of epoch {epoch}: {train_loss}")

                if not self.args.do_eval:
                    # if do not do evaluate, the checkpoint at the end of epoch needs to be saved
                    self.save_ck(model, epoch, tag="epoch")
            if self.use_tb:
                self._tb_writer.close()
