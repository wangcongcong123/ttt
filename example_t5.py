from ttt import *

if __name__ == '__main__':
    args = get_args()
    # check what args are available and their default values
    logger.info(f"args: {json.dumps(args.__dict__, indent=2)}")
    ############### customize args
    args.use_gpu = True
    # args.use_tpu = True
    args.do_train = True
    args.use_tb = True
    # any one from MODELS_SUPPORT (check:ttt/args.py)
    args.model_select = "t5-small"
    # select a dataset. First check if  it is from nlp, if yes load it here and save locally to the data_path
    # or customize a data in the data_path (train.json, val.json, test.json) where examples are organised in jsonl format
    # each line represents an example like this: {"text": "...", "label","..."}
    args.data_path = "data/glue/sst2"
    # any one from TASKS_SUPPORT (check:ttt/args.py)
    args.task = "t2t"
    args.log_steps = 400
    args.eval_batch_size=32
    args.per_device_train_batch_size=8

    # any one from LR_SCHEDULER_SUPPORT (check:ttt/args.py)
    args.scheduler = "warmuplinear"
    # set do_eval = False if your data does not contain a validation set. In that case, patience, and early_stop will be invalid
    args.do_eval = True
    args.tpu_address = "x.x.x.x"
    ############### end customize args
    # to have a sanity check for the args
    sanity_check(args)
    # seed everything, make deterministic
    set_seed(args.seed)
    tokenizer = get_tokenizer(args)
    inputs = get_inputs(tokenizer, args)
    model, strategy = create_model(args, logger, get_model)
    # start training, here we customize T2TTrainer to get more control and flexibility
    trainer = T2TTrainer(args)
    trainer.train(model, strategy, tokenizer, inputs)
