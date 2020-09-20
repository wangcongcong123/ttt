import math
from ttt import *

if __name__ == '__main__':
    args = get_args()
    model_path = "tmp/t5-large-fine-tuned-wnut-task3"
    args.output_path = model_path

    assert os.path.isdir(args.output_path)
    assert os.path.isfile(os.path.join(args.output_path, "args.json"))
    args = Args(**json.load(open(os.path.join(args.output_path, "args.json"))))
    set_name = "val"
    args.output_path = model_path
    out_name = set_name + "_preds"
    args.data_path = "data/final"
    args.use_gpu = True
    args.use_tpu = False
    args.eval_batch_size = 4
    model, strategy = create_model(args, logger, get_model, save_args=False)
    model.load_weights(args.output_path + "/tf_model.h5")

    tokenizer = AutoTokenizer.from_pretrained(args.output_path)
    logger.info(f"********************start predicting {set_name} set********************")
    source_texts, encoded_source, _, meta = convert_t2t_examples(
        os.path.join(args.data_path, f'{set_name}.json'), tokenizer, args, with_target=False, return_meta=True)
    source_input_ids = encoded_source["input_ids"]
    predict_dataset = tf.data.Dataset.from_tensor_slices(
        (source_input_ids, encoded_source["attention_mask"]))

    predict_dataset = predict_dataset.batch(args.eval_batch_size)
    iter_num = math.ceil(len(source_input_ids) / args.eval_batch_size)
    preds = []
    # with strategy.scope():
    for input_ids, attention_mask in tqdm(predict_dataset, total=iter_num, desc=f"predicting {set_name}..."):
        predicts = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  max_length=args.max_tgt_length)
        preds.extend([tokenizer.decode(ids) for ids in predicts])

    if not os.path.isdir("preds"):
        os.makedirs("preds")
    with open(f"preds/{out_name}.json", "w+") as out:
        for meta, source, target in zip(meta, source_texts, preds):
            meta["source"] = source
            meta["target"] = target
            out.write(json.dumps(meta) + "\n")

    print("done predicting")
