from ttt import *

if __name__ == '__main__':
    args = get_args()
    # check what args are available
    logger.info(f"args: {json.dumps(args.__dict__, indent=2)}")
    ############### customize args
    args.use_gpu = True
    # args.use_tpu = True
    # args.tpu_address = "x.x.x.x"  # replace with yours
    args.do_train = True
    args.use_tb = True
    # any one from MODELS_SUPPORT (check:ttt/args.py)
    args.model_select = "bert-base-uncased"
    # select a dataset following jsonl format, where text filed name is "text" and label field name is "label"
    args.data_path = "data/glue/sst2"
    # any one from TASKS_SUPPORT (check:ttt/args.py)
    args.task = "single-label-cls"
    args.log_steps = 400
    # any one from LR_SCHEDULER_SUPPORT (check:ttt/args.py)
    args.scheduler="warmuplinear"
    # set do_eval = False if your data does not contain a validation set. In that case, patience, and early_stop will be invalid
    args.do_eval = True
    ############### end customize args
    # to have a sanity check for the args
    sanity_check(args)
    # seed everything, make deterministic
    set_seed(args.seed)
    tokenizer = get_tokenizer(args)
    inputs = get_inputs(tokenizer, args)
    model, _ = create_model(args, logger, get_model)
    # start training, here we keras high-level API
    training_history = model.fit(
        inputs["x_train"],
        inputs["y_train"],
        epochs=args.num_epochs_train,
        verbose=2,
        batch_size=args.per_device_train_batch_size*args.num_replicas_in_sync,
        callbacks=get_callbacks(args, inputs, logger, get_evaluator),
    )