import torch
from ttt import *
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import transformers
from sklearn.metrics import accuracy_score, classification_report

transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()

def get_scheduler(optimizer, scheduler: str, warmup_steps: int, num_total: int):
    assert scheduler in ["constantlr", "warmuplinear", "warmupconstant", "warmupcosine",
                         "warmupcosinewithhardrestarts"], (
        'scheduler should be one of ["constantlr","warmupconstant","warmupcosine","warmupcosinewithhardrestarts"]')
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                               num_warmup_steps=warmup_steps,
                                                                               num_training_steps=num_total)

# we use transformers logger here so we can log message to a file locally
if __name__ == '__main__':
    args = get_args()
    ############### customize args
    # args.dataset_name = "c_data"
    # args.model_select = "t5-small-ex-pretrain"
    # args.from_pretrained = False

    args.dataset_name = "c_data"
    args.model_select = "t5-small"
    args.from_pretrained = True

    args.load_train_num = -1
    args.load_val_num = -1
    args.batch_size = 8
    args.epochs = 6
    args.log_steps = 400
    args.lr = 2e-4
    args.do_eval = True
    args.scheduler = "constantlr"
    # args.warmup_steps = 0.1
    args.grad_norm_clip = 1.0
    ##################
    add_filehandler_for_logger(".", logger)

    pyarrow_path = r"C:\Users\wangc\.cache\huggingface\datasets\{}\default\0.0.0".format(args.dataset_name)
    if not os.path.isdir(pyarrow_path):
        os.makedirs(pyarrow_path, exist_ok=True)

    dataset = load_dataset(f"{args.dataset_name}.py")

    if args.load_train_num > 0:
        train = load_dataset(f"{args.dataset_name}.py", split=f"train[:{args.load_train_num}]")
        dataset["train"] = train

    if args.load_val_num > 0:
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

    no_decay = ["bias", "LayerNorm.weight"]
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optim_groups = [
        {"params": params_decay, "weight_decay": 0.1},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(device)

    model.train().to(device)
    scheduler = get_scheduler(optimizer, scheduler=args.scheduler, warmup_steps=int(0.1 * len(train_dataloader)),
                              num_total=args.epochs * len(train_dataloader))

    def evaluate(val_dataloader):
        model.eval()
        gts = []
        preds = []
        for batch in tqdm(val_dataloader, total=len(val_dataloader), desc="evaluating..."):
            with torch.no_grad():
                batch.to(device)
                if isinstance(model,torch.nn.DataParallel):
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


    losses = []
    for epoch in tqdm(range(args.epochs), desc='epochs'):
        logger.info(f"start training epoch {epoch + 1}/{args.epochs}")
        base_steps = len(train_dataloader) * epoch
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, batch in pbar:
            batch.to(device)

            outputs = model(**batch, return_dict=True)
            loss = outputs.loss
            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            pbar.set_description(
                f"training - epoch {epoch + 1}/{args.epochs} iter {it}: train loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e}")

            if args.log_steps > 0 and (base_steps + it + 1) % args.log_steps == 0:
                logger.info(f"evaluate at global step = {base_steps + it + 1}")
                logger.info(f'Step {base_steps + it + 1} - mean train loss: {np.mean(losses):.3}')
                if args.do_eval:
                    evaluate()
                    model.train()

            if args.log_steps < 0:
                logger.info(f'Epoch {epoch + 1} - mean train loss: {np.mean(losses):.3}')
                logger.info(f"evaluate at epoch = {base_steps + it + 1}")
                if args.do_eval:
                    evaluate()
                    model.train()
