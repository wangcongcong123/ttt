
from ttt import *
from sklearn.metrics import classification_report, accuracy_score
import math

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # get args
    args = get_args()
    # check what args are available
    logger.info(f"args: {json.dumps(args.__dict__, indent=2)}")
    # args.do_train = True
    # args.do_eval = True
    # args.do_test = True
    # args.use_tpu = False
    # args.use_gpu = True
    sanity_check(args)

    if args.do_train:
        add_filehandler_for_logger(args.output_path, logger)
        logger.info(f"tf.version.VERSION: {tf.version.VERSION}")
        logger.info(f"set seed for random, numpy and tensorflow: seed = {args.seed}")
        set_seed(args.seed)
        tokenizer = get_tokenizer(args)
        inputs = get_inputs(tokenizer, args)
        model, strategy = create_model(args, logger, get_model)
        if not args.use_tpu:  # used during development
            model.run_eagerly = True  # for debugging, this is cool. TF also supports debugging as pytorch
        # start training here
        if args.task == "t2t":
            # t5-like model is customized because we want more flexibility and control over the training loop
            # customize_fit(model, strategy, tokenizer, inputs, args)
            trainer = T2TTrainer(args)
            trainer.train(model, strategy, tokenizer, inputs)
        else:
            training_history = model.fit(
                inputs["x_train"],
                inputs["y_train"],
                epochs=args.num_epochs_train,
                verbose=2,
                batch_size=args.per_device_train_batch_size*args.num_replicas_in_sync,
                callbacks=get_callbacks(args, inputs, logger, get_evaluator),
            )

    if args.do_test:
        add_filehandler_for_logger(args.output_path, logger, out_name="test")
        assert os.path.isdir(args.output_path)
        assert os.path.isfile(os.path.join(args.output_path, "args.json"))
        args = Args(**json.load(open(os.path.join(args.output_path, "args.json"))))
        model, _ = create_model(args,logger, get_model)
        # to make it ok to task == "single-label-cls" todo
        ck_path = glob.glob(os.path.join(args.output_path, "best*.h5"))[0]
        if args.ck_index_select < 0 and -args.ck_index_select <= args.keep_ck_num:
            cks_path_already = glob.glob(os.path.join(args.output_path, "*.h5"))
            index2path = {int(os.path.basename(each_ck_path).split(".")[0].split("_")[-1]): each_ck_path for
                          each_ck_path in cks_path_already}
            sorted_indices = sorted(index2path)
            ck_path = index2path[sorted_indices[args.ck_index_select]]

        logger.info(f"evaluating using weights from checkpoint: {ck_path}")
        model.load_weights(ck_path)

        tokenizer = AutoTokenizer.from_pretrained(args.output_path)

        logger.info("********************start evaluating on test set********************")
        if args.task == "single-label-cls":
            # tokenizer = get_tokenizer(args)
            test_texts, encoded_test, y_test = convert_seq_single_cls_examples(
                os.path.join(args.data_path, 'test.json'),
                tokenizer, args.input_seq_length,
                args.label2id)

            x_test = [encoded_test["input_ids"], encoded_test["token_type_ids"], encoded_test["attention_mask"]]

            if not args.use_tpu:  # used during development
                model.run_eagerly = True  # for debugging, this is cool. TF also supports debugging as pytorch

            pred_probs = model.predict(x_test, batch_size=32, steps=math.ceil(len(y_test) / 32), verbose=1)
            preds = tf.math.argmax(pred_probs, 1).numpy()

            acc = accuracy_score(y_test, preds)

            target_names = [''] * len(args.label2id)
            for label, id in args.label2id.items():
                target_names[id] = label

            logger.info(
                f"test_eval_report: {classification_report(y_test, preds, digits=4, target_names=target_names)}")
            logger.info(f"test_eval_acc: {acc}")

        elif args.task == "t2t":
            source_texts, encoded_source, encoded_target = convert_t2t_examples(
                os.path.join(args.data_path, 'test.json'), tokenizer, args.max_src_length, args.max_tgt_length)
            source_input_ids = encoded_source["input_ids"]
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (source_input_ids, encoded_source["attention_mask"], encoded_target["input_ids"]))
            test_dataset = test_dataset.batch(args.eval_batch_size)
            iter_num = math.ceil(len(source_input_ids) / args.eval_batch_size)
            preds = []
            gts = []

            for input_ids, attention_mask, gt in tqdm(test_dataset, total=iter_num, desc="testing..."):
                predicts = model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          max_length=args.max_tgt_length)
                preds.extend([tokenizer.decode(ids) for ids in predicts])
                gts.extend([tokenizer.decode(ids) for ids in gt])

            acc = accuracy_score(gts, preds)
            logger.info(
                f"test_eval_report: {classification_report(gts, preds, digits=4)}")
            logger.info(f"test_eval_acc: {acc}")
            # tensorboard dev upload --logdir runs