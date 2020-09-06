import json, os
from tqdm import tqdm
import numpy as np
import logging, pickle
from ttt.utils import add_filehandler_for_logger

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def read_seq_single_cls_examples(data_path):
    texts = []
    labels = []
    with open(data_path, "r") as f:
        for line in tqdm(f, desc=f"reading from {data_path}"):
            example = json.loads(line.strip())
            texts.append(example["text"])
            labels.append(example["label"])
    return texts, labels


def convert_seq_single_cls_examples(data_path, tokenizer, max_seq_length, label2id):
    texts, labels = read_seq_single_cls_examples(data_path)
    y = np.array([label2id[label] for label in labels])
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="np",
                              max_length=max_seq_length)
    return texts, encoded_texts, y


def prepare_seq_single_cls_inputs(tokenizer, args, load_train_num=-1):
    logger.info("reading train set")
    train_texts, train_labels = read_seq_single_cls_examples(os.path.join(args.data_path, "train.json"))

    if load_train_num > 0:
        assert load_train_num <= len(train_texts), f"there are {len(train_texts)} training examples"
        logger.info(f"loading only {load_train_num} training examples out of the totaling {len(train_texts)}")
        train_texts = train_texts[:load_train_num]
        train_labels = train_labels[:load_train_num]

    label2id = {}
    logger.info("building label2id from train examples")
    for i, label in enumerate(list(set(train_labels))):
        label2id[label] = i

    logger.info("converting labels to its ids for train examples")
    y_train = np.array([label2id[label] for label in train_labels])
    logger.info(f"encoding train examples (num={len(train_texts)})")
    logger.info(f"using tokenizer with padding = True and truncation = True and max_length = {args.max_seq_length}")
    logger.info("This may take a while")

    encoded_train = tokenizer(train_texts, padding=True, truncation=True, return_tensors="np",
                              max_length=args.max_seq_length)

    if "token_type_ids" not in encoded_train:
        # we need this for roberta tokenizer that does not return token_type_ids
        encoded_train["token_type_ids"] = np.zeros(encoded_train["input_ids"].shape, dtype=np.int32)

    x_train = [encoded_train["input_ids"], encoded_train["token_type_ids"], encoded_train["attention_mask"]]

    to_return = {"x_train": x_train, "y_train": y_train, "label2id": label2id}

    if os.path.isfile(os.path.join(args.data_path, "val.json")):
        logger.info(f"found validation set in {os.path.join(args.data_path, 'val.json')}")
        logger.info(f"encoding validation examples")
        val_texts, encoded_val, y_eval = convert_seq_single_cls_examples(os.path.join(args.data_path, 'val.json'),
                                                                         tokenizer, args.max_seq_length, label2id)
        if "token_type_ids" not in encoded_val:
            # we need this for roberta tokenizer that does not return token_type_ids
            encoded_val["token_type_ids"] = np.zeros(encoded_val["input_ids"].shape, dtype=np.int32)

        x_eval = [encoded_val["input_ids"], encoded_val["token_type_ids"], encoded_val["attention_mask"]]

        to_return.update({"x_eval": x_eval, "y_eval": y_eval, "eval_examples": val_texts})
    # if os.path.isfile(os.path.join(args.data_path, "test.json")):
    #     logger.info(f"found test set in {os.path.join(args.data_path, 'test.json')}")
    #     logger.info(f"encoding test examples")
    #     test_texts, encoded_test, y_test = convert_seq_single_cls_examples(os.path.join(args.data_path, 'test.json'),
    #                                                                        tokenizer, args.max_seq_length, label2id)
    #     x_test = [encoded_test["input_ids"], encoded_test["token_type_ids"], encoded_test["attention_mask"]]
    #     to_return.update({"x_test": x_test, "y_test": y_test, "test_examples": test_texts})
    return to_return


def read_t2t_examples(data_path, source_field_name="text", target_field_name="label", target_special_append_when_reading=" </s>"):
    source_texts = []
    target_texts = []
    with open(data_path, "r") as f:
        for line in tqdm(f, desc=f"reading from {data_path}"):
            example = json.loads(line.strip())
            source_texts.append(example[source_field_name])  # the </s> will be added after batch padding
            append_token=" </s>" if target_special_append_when_reading else ""
            target_texts.append(example[target_field_name] + append_token)
    return source_texts, target_texts


def convert_t2t_examples(data_path, tokenizer, args, append_token=" </s>"):
    source_texts, target_texts = read_t2t_examples(data_path, source_field_name=args.source_field_name,
                                                   target_field_name=args.target_field_name,
                                                   target_special_append_when_reading=args.target_special_append_when_reading)
    special_token_id = tokenizer.encode(append_token.strip())[0]

    encoded_source = tokenizer(source_texts, padding=True, truncation=True, return_tensors="np",
                               max_length=args.max_src_length)

    encoded_source["input_ids"], encoded_source["attention_mask"] = replace_with_special_token(encoded_source,
                                                                                               special_token_id)
    encoded_target = tokenizer(target_texts, padding=True, truncation=True, return_tensors="np",
                               max_length=args.max_tgt_length)
    if not args.target_special_append_when_reading:
        encoded_target["input_ids"], encoded_target["attention_mask"] = replace_with_special_token(encoded_target,
                                                                                               special_token_id)
    return source_texts, encoded_source, encoded_target


def replace_with_special_token(encoded, special_token_id, replace_token_id=0):
    # replace_token_id: padding id
    ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    # if np.all((ids == replace_token_id) == 0):
    #     new_ids = np.zeros((ids.shape[0], ids.shape[1] + 1), dtype=ids.dtype)
    #     new_ids[:, :1] = ids
    #     new_mask = np.zeros((attention_mask.shape[0], attention_mask.shape[1] + 1), dtype=attention_mask.dtype)
    #     new_mask[:, :1] = attention_mask
    #     return new_ids, new_mask

    for i in range(ids.shape[0]):
        indices = np.where(ids[i, :] == replace_token_id)[0]
        if indices.size == 0:
            ids[i, -1] = special_token_id
        else:
            ids[i, indices[0]] = special_token_id
            attention_mask[i, indices[0]] = 1
    return ids, attention_mask


def shift_to_right(input_ids, decoder_start_token_id):
    shifted_input_ids = np.zeros(input_ids.shape, dtype=input_ids.dtype)
    shifted_input_ids[..., 1:] = input_ids[..., :-1]
    shifted_input_ids[..., 0] = decoder_start_token_id
    return shifted_input_ids


def prepare_t2t_inputs(tokenizer, args, load_train_num=-1):
    logger.info("reading train set")
    source_texts_train, target_texts_train = read_t2t_examples(os.path.join(args.data_path, "train.json"),
                                                               source_field_name=args.source_field_name,
                                                               target_field_name=args.target_field_name,
                                                               target_special_append_when_reading=args.target_special_append_when_reading)

    if load_train_num > 0:
        assert load_train_num <= len(source_texts_train), f"there are {len(source_texts_train)} training examples"
        logger.info(f"loading only {load_train_num} training examples out of the totaling {len(source_texts_train)}")
        source_texts_train = source_texts_train[:load_train_num]
        target_texts_train = target_texts_train[:load_train_num]

    logger.info(f"encoding source train examples (num={len(source_texts_train)})")
    logger.info(f"using tokenizer with padding = True and truncation = True and max_src_length = {args.max_src_length}")
    logger.info("This may take a while")
    encoded_source_train = tokenizer(source_texts_train, padding=True, truncation=True, return_tensors="np",
                                     max_length=args.max_src_length)

    logger.info(f"encoding target train examples (num={len(target_texts_train)})")
    logger.info(f"using tokenizer with padding = True and truncation = True and max_tgt_length = {args.max_tgt_length}")
    logger.info("This may take a while")
    encoded_target_train = tokenizer(target_texts_train, padding=True, truncation=True, return_tensors="np",
                                     max_length=args.max_tgt_length)

    special_token_id = tokenizer.encode("</s>")[0]
    decoder_start_token_id = 0  # hard coded here, todo

    # add "</s>" add the end of each sequence for source, it already appends to the target sequences when reading from file
    train_source_input_ids, train_source_attention_mask = replace_with_special_token(encoded_source_train,
                                                                                     special_token_id)
    if not args.target_special_append_when_reading:
        train_target_input_ids, train_target_attention_mask = replace_with_special_token(encoded_target_train,
                                                                                         special_token_id)
    else:
        train_target_input_ids, train_target_attention_mask = encoded_target_train["input_ids"], encoded_target_train[
            "attention_mask"]
    # this is pytorch's cross entropy's ignore index. to  figure out this in tensorflow-2.0
    # train_target_input_ids[train_target_input_ids == 0] = -100
    x_train = [train_source_input_ids, train_source_attention_mask,
               shift_to_right(train_target_input_ids, decoder_start_token_id), train_target_attention_mask]

    to_return = {"x_train": x_train, "y_train": train_target_input_ids}
    if args.do_eval:
        assert os.path.isfile(
            os.path.join(args.data_path, "val.json")), "do_eval=True, and no validation data (val.json) is found"

    if os.path.isfile(os.path.join(args.data_path, "val.json")):
        logger.info(f"found validation set in {os.path.join(args.data_path, 'val.json')}")
        logger.info(f"encoding validation examples")
        source_texts, encoded_source, encoded_target = convert_t2t_examples(
            os.path.join(args.data_path, 'val.json'),
            tokenizer, args)
        # add "</s>" add the end of each sequence
        source_input_ids, source_attention_mask = encoded_source["input_ids"], encoded_source["attention_mask"]
        target_input_ids, target_attention_mask = encoded_target["input_ids"], encoded_target["attention_mask"]
        x_eval = [source_input_ids, source_attention_mask, shift_to_right(target_input_ids, decoder_start_token_id),
                  target_attention_mask]
        # this is pytorch's cross entropy's ignore index. to  figure out this in tensorflow-2.0
        # target_input_ids[target_input_ids == 0] = -100
        to_return.update({"x_eval": x_eval, "y_eval": target_input_ids, "eval_examples": source_texts})
    # if os.path.isfile(os.path.join(args.data_path, "test.json")):
    #     logger.info(f"found test set in {os.path.join(args.data_path, 'test.json')}")
    #     logger.info(f"encoding test examples")
    #     test_texts, encoded_test, y_test = convert_seq_single_cls_examples(os.path.join(args.data_path, 'test.json'),
    #                                                                        tokenizer, args.max_seq_length, label2id)
    #     x_test = [encoded_test["input_ids"], encoded_test["token_type_ids"], encoded_test["attention_mask"]]
    #     to_return.update({"x_test": x_test, "y_test": y_test, "test_examples": test_texts})
    return to_return


def get_with_prepare_func(tokenizer, args, prepare_func, load_train_num=-1, is_cache=False):
    '''
        :param tokenizer:
        :param args:
        :return:
    '''
    args.is_data_cache = is_cache
    if is_cache:
        if load_train_num > 0:
            data_cache_path = os.path.join(args.data_path,
                                           f"{args.model_select.replace('/', '-')}-data-{load_train_num}.pkl")
        else:
            data_cache_path = os.path.join(args.data_path, f"{args.model_select.replace('/', '-')}-data.pkl")
        args.data_cache_path = data_cache_path
        if os.path.isfile(data_cache_path):
            with open(data_cache_path, "rb") as f:
                logger.info(f"reading cached data from {data_cache_path}")
                logger.warning(
                    f"if you changed the max_seq_length/max_src_length/max_tgt_length, this may not correctly loaded, since the {data_cache_path} is pickled based on first time loading")
                to_return = pickle.load(f)
        else:
            to_return = prepare_func(tokenizer, args, load_train_num=load_train_num)
            with open(data_cache_path, "wb") as f:
                logger.info(f"caching data to {data_cache_path}")
                pickle.dump(to_return, f)
    else:
        to_return = prepare_func(tokenizer, args, load_train_num=load_train_num)
    return to_return


def get_inputs(tokenizer, args):
    add_filehandler_for_logger(args.output_path, logger)
    if args.task == "single-label-cls":
        inputs = get_with_prepare_func(tokenizer, args, prepare_seq_single_cls_inputs, is_cache=True)
        args.input_seq_length = inputs["x_train"][0].shape[-1]
        args.label2id = inputs["label2id"]
        return inputs
    elif args.task == "t2t" or args.task == "translation":
        args.target_special_append_when_reading = args.task == "t2t"
        data_dict = get_with_prepare_func(tokenizer, args, prepare_t2t_inputs, is_cache=True)
        args.source_sequence_length = data_dict["x_train"][0].shape[-1]
        args.target_sequence_length = data_dict["x_train"][2].shape[-1]
        return data_dict
    else:
        # when more tasks are supported -> todo
        pass
