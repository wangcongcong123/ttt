from transformers import TFAutoModelWithLMHead, AutoTokenizer
from datasets import load_dataset, load_metric
import json, os
from tqdm import tqdm
import tensorflow as tf
import transformers

transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()

from ttt import check_output_path, dictionize_single_dataset, save_and_check_if_early_stop, get_args, T2TTrainer, get_strategy, add_filehandler_for_logger, get_existing_cks


def get_dataset(data, tag="train", return_raw_inputs=False):
    actual_max_src_length = data["source_lengths"].numpy().max()
    actual_max_tgt_length = data["target_lengths"].numpy().max()
    logger.info(f"actual_max_src_length ({tag}) = {actual_max_src_length}")
    logger.info(f"actual_max_tgt_length ({tag}) = {actual_max_tgt_length}")
    features = {
        x: data[x].to_tensor(default_value=tokenizer.pad_token_id, shape=[None, actual_max_src_length]) for
        x in ['input_ids', 'attention_mask']}  # padding here, in memory padding
    features.update({"decoder_attention_mask": data["decoder_attention_mask"].to_tensor(
        default_value=tokenizer.pad_token_id, shape=[None, actual_max_tgt_length])})
    raw_inputs = (features, data["labels"].to_tensor(default_value=tokenizer.pad_token_id, shape=[None, actual_max_tgt_length]))
    # there are some compability concerns here, we rename the input names here to be consistent with T2TTrainer.train()
    tmp_inputs = dictionize_single_dataset(raw_inputs, tag=tag)
    x, y = tmp_inputs[f"x_{tag}"], tmp_inputs[f"y_{tag}"]
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if return_raw_inputs:
        return dataset, raw_inputs
    return dataset


def convert_to_features(example_batch, args, tokenizer):
    encoded_source = tokenizer(example_batch["source"], padding=True, truncation=True,
                               max_length=args.max_src_length)
    encoded_target = tokenizer(example_batch["target"], padding=True, truncation=True,
                               max_length=args.max_tgt_length)
    source_lengths = [len(encoded_source["input_ids"][0])] * len(encoded_source["input_ids"])
    target_lengths = [len(encoded_target["input_ids"][0])] * len(encoded_target["input_ids"])

    encoded_source.update(
        {"labels": encoded_target["input_ids"], "source_lengths": source_lengths, "target_lengths": target_lengths,
         "decoder_attention_mask": encoded_target["attention_mask"]})
    return encoded_source


def evaluate(args, logger, model, tokenizer, eval_dataset, steps=0, tag="epoch", is_test=False, eval_length=None):
    gts = []
    preds = []
    if eval_length is not None:
        eval_steps = eval_length
    else:
        eval_steps = tf.data.experimental.cardinality(eval_dataset).numpy()
    logger.info(f"start evaluating at {tag}={steps}")
    for inputs, labels in tqdm(eval_dataset, total=eval_steps, desc="evaluating..."):
        predictions = model.generate(input_ids=inputs["source_input_ids"],
                                     attention_mask=inputs["source_attention_mask"],
                                     max_length=args.max_tgt_length)
        pred = [tokenizer.decode(ids) for ids in predictions]
        gt = [tokenizer.decode(ids) for ids in labels["target_input_ids"]]
        preds.extend(pred)
        gts.extend(gt)

    metrics_fn = load_metric("cls_metric.py", "short")
    metrics = metrics_fn.compute(predictions=preds, references=gts)

    logger.info(f"val_cls_report: {json.dumps(metrics, indent=2)}")
    eval_score = metrics[args.eval_on]
    logger.info(f"val_{args.eval_on}_score: {eval_score}")

    is_early_stop = False

    if not is_test:
        is_early_stop = save_and_check_if_early_stop(eval_score, args, logger, model, tokenizer, steps=steps, tag=tag, from_tf=True)

    return {"eval_scores": metrics, "preds": preds, "is_early_stop": is_early_stop}


if __name__ == '__main__':
    args = get_args()
    # check what args are available and their default values
    logger.info(f"args: {json.dumps(args.__dict__, indent=2)}")
    ############### customize args
    args.use_gpu = True
    # args.use_tpu = True
    # args.tpu_address = "x.x.x.x"
    # use tensorboard for logging
    args.use_tb = True

    # model configuration
    args.model_select = "t5-small"
    args.max_src_length = 256
    args.max_tgt_length = 10
    args.per_device_train_batch_size = 16
    args.eval_batch_size = 32
    # load data from a customized data loading script
    args.dataset_name = "covid_data.py, default"
    # any one from TASKS_SUPPORT (check:ttt/args.py)
    args.log_steps = 400
    args.eval_batch_size = 32
    args.per_device_train_batch_size = 8

    # any one from LR_SCHEDULER_SUPPORT (check:ttt/args.py)
    args.scheduler = "warmuplinear"
    args.lr = 5e-5
    # use tf.keras.optimizers.Adam optimizer by default in train()

    args.do_train = True
    args.do_eval = True
    args.do_test = True

    # what to evaluated if the validation set and an evaluation callback function are passed to the T2TTrainer's train method
    args.eval_on = "acc"
    # how many checkpoints to keep based on args.log_steps if args.do_train = True
    args.keep_ck_num = 3
    # use the best on validation set as the checkpoint on test set evaluation if args.do_test = True
    args.ck_index_select = 0
    ############### end customize args
    # construct the output path argument to save everything to this path
    args.output_path = os.path.join("tmp", f"{args.model_select}_covid_info")
    check_output_path(args.output_path, force=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_select)
    dataset = load_dataset(*args.dataset_name.split(", "))
    # use num_proc = 6 can give 6x speedup ideally as compared to 1 proc, which is really good stuff for tokenizing many examples
    # this is the main reason why using HF's datasets instead of torch.Dataset
    encoded = dataset.map(convert_to_features, batched=True, fn_kwargs={"args": args, "tokenizer": tokenizer}, num_proc=6)
    columns = ['input_ids', "source_lengths", "target_lengths", 'attention_mask', 'labels', 'decoder_attention_mask']
    encoded.set_format(type='tensorflow', columns=columns)

    if args.do_train:
        add_filehandler_for_logger(args.output_path, logger, out_name="train")
        strategy = get_strategy(args, logger)
        with strategy.scope():
            # from_pt to aovid repeated downloading
            model = TFAutoModelWithLMHead.from_pretrained(args.model_select, from_pt=True)
            train_dataset = get_dataset(encoded["train"], tag="train")
            val_dataset = None
            if "validation" in encoded:
                val_dataset = get_dataset(encoded["validation"], tag="eval")
            trainer = T2TTrainer(args, logger)
            trainer.train(model, strategy, tokenizer, train_dataset=train_dataset, eval_dataset=val_dataset, evaluate_fn=evaluate, verbose=True)

    # we want the testing is independent of the training as much as possible
    # so that it is okay to do test when args.do_train = False and checkpoints already exist
    if args.do_test:
        test_set = "test"
        if test_set in encoded:
            add_filehandler_for_logger(args.output_path, logger, out_name="test")
            sorted_indices, index2path = get_existing_cks(args.output_path, return_best_ck=False)
            if args.ck_index_select < 0:
                model_path = index2path[sorted_indices[args.ck_index_select]]
            else:
                bests = [name for name in os.listdir(args.output_path) if name.startswith("best")]
                if bests != []:
                    model_path = os.path.join(args.output_path, bests[0])
                else:
                    model_path = index2path[sorted_indices[args.ck_index_select]]
            model = TFAutoModelWithLMHead.from_pretrained(model_path)
            logger.info(f"-------------------eval and predict on {test_set} set-------------------")
            test_dataset = get_dataset(encoded[test_set])
            test_dataset = test_dataset.batch(args.eval_batch_size)
            eval_dict = evaluate(args, logger, model, tokenizer, test_dataset, is_test=True)
        else:
            raise ValueError(f"Not found {test_set} for evaluation")