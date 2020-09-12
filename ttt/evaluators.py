''''
this is a evaluation callback class for high-level Keras training (BERT-like models in this lib)
'''

import sys
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import logging
import os
from ttt.utils import add_filehandler_for_logger, get_existing_cks
from tensorboardX import SummaryWriter

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ClsEvaluator(keras.callbacks.Callback):
    def __init__(self, x_eval, y_eval, args):
        super(ClsEvaluator).__init__()
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.eval_on = args.eval_on
        self.patience = args.patience
        self.log_steps = args.log_steps
        self.args = args
        self.use_tb = self.args.use_tb
        if self.use_tb:
            self._tb_writer = SummaryWriter(log_dir=os.path.join("runs", args.output_folder))

    def on_train_begin(self, logs=None):
        self.global_step = 0
        self.wait = 0
        self.best = np.Inf if self.eval_on == "loss" else -np.Inf

    def on_train_end(self, logs=None):
        if self.use_tb:
            self._tb_writer.close()

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        if self.log_steps != -1 and self.global_step % self.log_steps == 0:
            if self.args.do_eval:
                self.evaluate(self.global_step, tag="global_step", logs=logs)

    def evaluate(self, steps, tag="epoch", logs=None):
        logger.info("\n")
        logger.info(f"*************evaluating at {tag}={steps}*************")
        eval_results = self.model.evaluate(self.x_eval, self.y_eval, batch_size=self.args.eval_batch_size)
        dev_loss, acc = eval_results[0], eval_results[1]
        pred_probs = self.model.predict(self.x_eval, batch_size=self.args.eval_batch_size)
        preds = tf.math.argmax(pred_probs, 1).numpy()
        # acc = accuracy_score(preds, self.y_eval)
        logger.info(f"{tag}={steps}, eval_report: {classification_report(self.y_eval, preds, digits=4)}")
        logger.info(f"{tag}={steps}, eval_acc: {acc}")
        logger.info(f"{tag}={steps}, eval_results: {eval_results}")
        logger.info(f"{tag}={steps}, train_logs: {logs}")

        if self.use_tb:
            if logs is not None:
                logger.info("logging metrics with tensorboard")
                for key,value in logs.items():
                    self._tb_writer.add_scalar(f"train_{key}_{tag}",value, steps)
                self._tb_writer.add_scalar(f"train_lr_{tag}", self.model.optimizer.lr.numpy(), steps)
                self._tb_writer.add_scalar(f"val_acc_{tag}", acc, steps)
                self._tb_writer.add_scalar(f"val_loss_{tag}", dev_loss, steps)


        if self.eval_on == "loss":
            if dev_loss <= self.best:
                self.wait = 0
                self.best = dev_loss
                self.save_ck(steps, tag, best_ck=True)
            else:
                self.wait += 1
        else:
            if acc >= self.best:
                self.wait = 0
                self.best = acc
                self.save_ck(steps, tag, best_ck=True)
            else:
                self.wait += 1
        logger.info(f"early stop count: {self.wait}/{self.patience}")
        logger.info(f"{tag}={steps}, best_on_eval_since({self.eval_on}): {self.best}")
        self.save_ck(steps, tag)
        if self.wait >= self.patience:
            logger.info("run out of patience, early stop")
            if self.use_tb:
                self._tb_writer.close()
            sys.exit(0)

    def save_ck(self, steps, tag="epoch", best_ck=False):
        sorted_indices, index2path = get_existing_cks(self.args.output_path, best_ck=best_ck)
        if len(sorted_indices) >= self.args.keep_ck_num:
            logger.info(
                f"since there are already {len(sorted_indices)} checkpoints saved that will be more than keep_ck_num={self.args.keep_ck_num}")
            logger.info(f"remove the oldest one: {index2path[sorted_indices[0]]}")
            os.remove(index2path[sorted_indices[
                0]])  # remove the oldest checkpoint, i.e., the one with the lowest epoch number
        # write_args(self.args.output_path, self.args)
        if best_ck:
            logger.info(
                f'save best model weights to {os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5")}')
            self.model.save_weights(os.path.join(self.args.output_path, f"best_ck_at_{tag}_{steps}.h5"),
                                    overwrite=True)
        else:
            logger.info(
                f'save model weights to {os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5")}')
            self.model.save_weights(os.path.join(self.args.output_path, f"ck_at_{tag}_{steps}.h5"),
                                    overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.log_steps == -1:
            if self.args.do_eval:
                self.evaluate(epoch, tag="epoch",logs=logs)
        if not self.args.do_eval:
            # if do not do evaluate, the checkpoint at the end of epoch needs to be saved
            self.save_ck(epoch, tag="epoch")

def get_evaluator(x_eval, y_eval, args):
    add_filehandler_for_logger(args.output_path, logger)
    if args.task == "single-label-cls":
        return ClsEvaluator(x_eval, y_eval, args)
    elif args.task == "t2t":
        # it uses customize training toop, we do not need a evaluator here
        pass
    else:
        pass