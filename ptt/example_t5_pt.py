import torch
from ttt import *
import sys, os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import transformers
from sklearn.metrics import accuracy_score, classification_report

transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()

def set_seed(seed, n_gpu):
    logger.info(f"   see seed for random, numpy and torch {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def evaluate(model, eval_dataloader):
    model.eval()
    gts = []
    preds = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="evaluating..."):
        with torch.no_grad():
            batch.to(device)
            if isinstance(model, torch.nn.DataParallel):
                predictions = model.module.generate(input_ids=batch["input_ids"],
                                                    attention_mask=batch["attention_mask"])
            else:
                predictions = model.generate(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"])
            pred = [tokenizer.decode(ids) for ids in predictions]
            gt = [tokenizer.decode(ids) for ids in batch["labels"]]
            preds.extend(pred)
            gts.extend(gt)
    eval_score = accuracy_score(gts, preds)
    logger.info(f"val_eval_score: {eval_score}")
    logger.info(f"val_cls_report: {classification_report(gts, preds, digits=4)}")


# we use transformers logger here so we can log message to a file locally
if __name__ == '__main__':
    args = get_args()
    ############### customize args
    args.dataset_name = "sst"
    pyarrow_path = r"C:\Users\wangc\.cache\huggingface\datasets\{}\default\0.0.0".format(args.dataset_name)

    if not sys.platform.startswith("win"):
        pyarrow_path = f"/home/congcong/.cache/huggingface/datasets/{args.dataset_name}/default/0.0.0"

    if not os.path.isdir(pyarrow_path):
        os.makedirs(pyarrow_path, exist_ok=True)

    dataset = load_dataset(f"data_scripts/{args.dataset_name}.py")

    args.model_select = "t5-small"
    args.from_pretrained = True
    args.batch_size = 8
    args.epochs = 1
    args.log_steps = 400
    args.lr = 2e-4
    args.do_eval = True
    args.scheduler = "constantlr"
    # args.warmup_steps = 0.1
    args.grad_norm_clip = 1.0
    args.do_test = True
    args.output_path = os.path.join("tmp", args.model_select + "_" + args.dataset_name)
    ##################
    set_seed(args.seed, torch.cuda.device_count())
    check_output_path(args.output_path, force=True)
    add_filehandler_for_logger(args.output_path, logger)

    if hasattr(args, "load_train_num") and args.load_train_num > 0:
        train = load_dataset(f"{args.dataset_name}.py", split=f"train[:{args.load_train_num}]")
        dataset["train"] = train

    if hasattr(args, "load_val_num") and args.load_val_num > 0:
        val = load_dataset(f"{args.dataset_name}.py", split=f"validation[:{args.load_val_num}]")
        dataset["validation"] = val

    tokenizer = T5Tokenizer.from_pretrained(args.model_select)

    def convert_to_features(example_batch):
        encoded_source = tokenizer(example_batch["source"])
        encoded_target = tokenizer(example_batch["target"])
        encoded_source.update(
            {"labels": encoded_target["input_ids"], "decoder_attention_mask": encoded_target["attention_mask"]})
        return encoded_source


    def collate_fn(examples):
        source_inputs = [{"input_ids": each["input_ids"], "attention_mask": each["attention_mask"]} for each in
                         examples]
        target_inputs = [{"input_ids": each["labels"], "attention_mask": each["decoder_attention_mask"]} for each in
                         examples]
        source_inputs_padded = tokenizer.pad(source_inputs, return_tensors='pt')
        target_inputs_padded = tokenizer.pad(target_inputs, return_tensors='pt')
        source_inputs_padded.update({"labels": target_inputs_padded["input_ids"],
                                     "decoder_attention_mask": target_inputs_padded["attention_mask"]})
        return source_inputs_padded


    encoded = dataset.map(convert_to_features, batched=True)
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    encoded.set_format(type='torch', columns=columns)

    train_dataloader = torch.utils.data.DataLoader(encoded["train"], collate_fn=collate_fn, batch_size=args.batch_size)
    val_dataloader = torch.utils.data.DataLoader(encoded["validation"], collate_fn=collate_fn,
                                                 batch_size=args.batch_size * 4)
    if args.from_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(args.model_select)
    else:
        config = T5Config.from_pretrained(args.model_select)
        model = T5ForConditionalGeneration(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from src.trainer import train

    train(args, logger, model, device, train_dataloader, val_dataloader=val_dataloader, evaluate=evaluate)
    # save model training details to output path at the end of model training. todo -> save based on args.keep_ck_num and best save on val set evaluation within training loop
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    write_args(args.output_path, args)

    if args.do_test:
        logger.info("-------------------eval on test set-------------------")
        test_dataloader = torch.utils.data.DataLoader(encoded["test"], collate_fn=collate_fn,
                                                      batch_size=args.batch_size * 4)
        evaluate(model, test_dataloader)
