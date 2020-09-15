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


def prepare_seq_single_cls_inputs(tokenizer, args, load_train_num=-1, tokenize_batch_size=None):
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
    # todo -> tokenize_batch_size
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


def read_t2t_examples(data_path, source_field_name="text", target_field_name="label", with_target=True):
    source_texts = []
    target_texts = []
    other_attributes = []
    with open(data_path, "r") as f:
        for line in tqdm(f, desc=f"reading from {data_path}"):
            example = json.loads(line.strip())

            source_texts.append(example.pop(source_field_name))
            # the </s> will be added when calling tokenizer, automatically when its add_special_tokens=True
            if with_target:
                target_texts.append(example.pop(target_field_name))
            other_attributes.append(example)
    return source_texts, target_texts, other_attributes


def convert_t2t_examples(data_path, tokenizer, args, with_target=True, return_meta=False):
    source_texts, target_texts, other_attributes = read_t2t_examples(data_path,
                                                                     source_field_name=args.source_field_name,
                                                                     target_field_name=args.target_field_name,
                                                                     with_target=with_target)

    # for pre - training t5, no need to append the </s> to the source sequence
    encoded_source = tokenizer(source_texts, padding=True, truncation=True, return_tensors="np",
                               max_length=args.max_src_length, add_special_tokens=not args.is_pretrain)

    encoded_target = None
    if with_target:
        # for making predictions purpose
        encoded_target = tokenizer(target_texts, padding=True, truncation=True, return_tensors="np",
                                   max_length=args.max_tgt_length, add_special_tokens=True)
    if return_meta:
        return source_texts, encoded_source, encoded_target, other_attributes

    return source_texts, encoded_source, encoded_target


def replace_with_special_token(encoded, special_token_id, replace_token_id=0):
    # replace_token_id: padding id
    ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

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


def tokenize_with_progress_bar(tokenizer, args, text_list, batch_size=1000):
    assert batch_size > 0
    encoded_return = {"input_ids": [], "attention_mask": []}
    batch = []
    # actual_max_seq = =args.max_src_length
    # tokenization in batch specified by batch_size, a bit different padding here compared to the at-one-go way, the batch here always pad to max_src_length although the longest one is not that long
    for idx, each_text in tqdm(enumerate(text_list), desc="tokenizing by batch...", total=len(text_list)):
        if (idx + 1) % batch_size == 0:
            batch.append(each_text)
            encoded = tokenizer(batch, padding="max_length", truncation=True, max_length=args.max_src_length,
                                add_special_tokens=not args.is_pretrain)
            encoded_return["input_ids"].extend(encoded["input_ids"])
            encoded_return["attention_mask"].extend(encoded["attention_mask"])
            batch = []
        else:
            batch.append(each_text)

    if batch != []:
        encoded = tokenizer(batch, padding="max_length", truncation=True, max_length=args.max_src_length,
                            add_special_tokens=not args.is_pretrain)
        encoded_return["input_ids"].extend(encoded["input_ids"])
        encoded_return["attention_mask"].extend(encoded["attention_mask"])

    assert len(encoded_return["input_ids"]) == len(text_list)

    encoded_return["input_ids"] = np.array(encoded_return["input_ids"])

    assert len(encoded_return["attention_mask"]) == len(text_list)

    encoded_return["attention_mask"] = np.array(encoded_return["attention_mask"])

    from transformers import BatchEncoding
    return BatchEncoding(data=encoded_return)


def tokenizet2t(tokenizer, args, source_text, target_text, batch_size=None):
    # this method is similar to T5Tokenizer.prepare_seq2seq_batch, it is rewritten here to get more control and flexibility
    logger.info(f"encoding source examples (num={len(source_text)})")
    logger.info(f"using tokenizer with padding = True and truncation = True and max_src_length = {args.max_src_length}")

    if batch_size is None:
        # if batch_size is None. refers to tokenization at one go. In this case, no progress bar but it can save memory potentially
        # for pre - training t5, no need to append the </s> to the source sequence
        logger.info("The tokenization is conducted at the backend without progress bar")
        logger.info("This may take a while")
        encoded_source = tokenizer(source_text, padding=True, truncation=True, return_tensors="np",
                                   max_length=args.max_src_length - 1 if not args.is_pretrain else args.max_src_length,
                                   add_special_tokens=not args.is_pretrain)
    else:
        encoded_source = tokenize_with_progress_bar(tokenizer, args, source_text, batch_size=batch_size)

    logger.info(f"encoding target examples (num={len(target_text)})")
    logger.info(f"using tokenizer with padding = True and truncation = True and max_tgt_length = {args.max_tgt_length}")
    if batch_size is None:
        logger.info("The tokenization is conducted at the backend without progress bar")
        logger.info("This may take a while")
        encoded_target = tokenizer(target_text, padding=True, truncation=True, return_tensors="np",
                                   max_length=args.max_tgt_length, add_special_tokens=True)
    else:
        encoded_target = tokenize_with_progress_bar(tokenizer, args, target_text, batch_size=batch_size)
    return encoded_source, encoded_target


def prepare_t2t_inputs(tokenizer, args, load_train_num=-1, tokenize_batch_size=None):
    logger.info("reading train set")
    source_texts_train, target_texts_train, _ = read_t2t_examples(os.path.join(args.data_path, "train.json"),
                                                                  source_field_name=args.source_field_name,
                                                                  target_field_name=args.target_field_name)
    if load_train_num > 0:
        assert load_train_num <= len(source_texts_train), f"there are {len(source_texts_train)} training examples"
        logger.info(f"loading only {load_train_num} training examples out of the totaling {len(source_texts_train)}")
        source_texts_train = source_texts_train[:load_train_num]
        target_texts_train = target_texts_train[:load_train_num]

    encoded_source_train, encoded_target_train = tokenizet2t(tokenizer, args, source_texts_train, target_texts_train,
                                                             batch_size=tokenize_batch_size)

    # special_token_id = tokenizer.eos_token_id
    decoder_start_token_id = tokenizer.pad_token_id

    train_source_input_ids, train_source_attention_mask = encoded_source_train["input_ids"], encoded_source_train[
        "attention_mask"]

    train_target_input_ids, train_target_attention_mask = encoded_target_train["input_ids"], encoded_target_train[
        "attention_mask"]
    # this is pytorch's cross entropy's ignore index. to  figure out this in tensorflow-2.0
    # -> update: this can be found in transformers.modeling_tf_utils:TFCausalLanguageModelingLoss (since transformers 3.1.0)
    # shift_to_right(train_target_input_ids, decoder_start_token_id), this not needed since transformers 3.1.0 (it automatically takes care of these by simply passing labels argument to the model, just like in pytorch)
    # x_train = [train_source_input_ids, train_source_attention_mask,
    #            shift_to_right(train_target_input_ids, decoder_start_token_id), train_target_attention_mask]
    x_train = {"source_input_ids":train_source_input_ids, "source_attention_mask":train_source_attention_mask,
               "shifted_target_input_ids":shift_to_right(train_target_input_ids, decoder_start_token_id), "target_attention_mask":train_target_attention_mask}
    train_target_input_ids[train_target_input_ids == 0] = -100
    to_return = {"x_train": x_train, "y_train": {"target_input_ids":train_target_input_ids}}

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
        # target_input_ids[target_input_ids == 0] = -100
        # in t5 trainer, eval lm labels are not used for calculating loss but for generation comparison, os we do not apply -100 replacement here

        # x_eval = [source_input_ids, source_attention_mask, shift_to_right(target_input_ids, decoder_start_token_id),
        #           target_attention_mask]
        x_eval = {"source_input_ids": source_input_ids, "source_attention_mask": source_attention_mask,
                   "shifted_target_input_ids": shift_to_right(target_input_ids, decoder_start_token_id),
                   "target_attention_mask": target_attention_mask}
        to_return.update({"x_eval": x_eval, "y_eval": {"target_input_ids":target_input_ids}, "eval_examples": {"source_texts":source_texts}})

    # if os.path.isfile(os.path.join(args.data_path, "test.json")):
    #     logger.info(f"found test set in {os.path.join(args.data_path, 'test.json')}")
    #     logger.info(f"encoding test examples")
    #     source_texts, encoded_source, encoded_target = convert_t2t_examples(
    #         os.path.join(args.data_path, 'test.json'),
    #         tokenizer, args)
    #     # add "</s>" add the end of each sequence
    #     source_input_ids, source_attention_mask = encoded_source["input_ids"], encoded_source["attention_mask"]
    #     target_input_ids, target_attention_mask = encoded_target["input_ids"], encoded_target["attention_mask"]
    #     # target_input_ids[target_input_ids == 0] = -100
    #     # in t5 trainer, eval lm labels are not used for calculating loss but for generation comparison, os we do not apply -100 replacement here
    #     x_test = [source_input_ids, source_attention_mask, shift_to_right(target_input_ids, decoder_start_token_id),
    #               target_attention_mask]
    #     to_return.update({"x_test": x_test, "y_test": target_input_ids, "test_examples": source_texts})
    return to_return


def get_with_prepare_func(tokenizer, args, prepare_func, load_train_num=-1, check_cache=False,
                          tokenize_batch_size=None):
    '''
        :param tokenizer:
        :param args:
        :return:
    '''
    args.is_load_from_data_cache = False
    if check_cache:
        if load_train_num > 0:
            data_cache_path = os.path.join(args.data_path,
                                           f"{args.model_select.replace('/', '-')}-data-{load_train_num}.pkl")
        else:
            data_cache_path = os.path.join(args.data_path, f"{args.model_select.replace('/', '-')}-data.pkl")
        args.data_cache_path = data_cache_path
        if os.path.isfile(data_cache_path):
            args.is_load_from_data_cache = True
            with open(data_cache_path, "rb") as f:
                logger.info(f"reading cached data from {data_cache_path}")
                logger.warning(
                    f"if you changed the max_seq_length/max_src_length/max_tgt_length, this may not correctly loaded, since the {data_cache_path} is pickled based on first time loading")
                to_return = pickle.load(f)
        else:
            to_return = prepare_func(tokenizer, args, load_train_num=load_train_num,
                                     tokenize_batch_size=tokenize_batch_size)
            with open(data_cache_path, "wb") as f:
                logger.info(f"caching data to {data_cache_path}")
                pickle.dump(to_return, f)
    else:
        to_return = prepare_func(tokenizer, args, load_train_num=load_train_num)
    return to_return

def get_inputs(tokenizer, args):
    add_filehandler_for_logger(args.output_path, logger)
    if args.task == "single-label-cls":
        inputs = get_with_prepare_func(tokenizer, args, prepare_seq_single_cls_inputs, check_cache=True,
                                       load_train_num=100000)
        args.input_seq_length = inputs["x_train"][0].shape[-1]
        args.label2id = inputs["label2id"]
        return inputs
    elif args.task == "t2t" or args.task == "translation" or args.task == "pretrain":
        args.is_pretrain = args.task == "pretrain"
        tokenize_batch_size = 1000 if args.task == "pretrain" else None
        data_dict = get_with_prepare_func(tokenizer, args, prepare_t2t_inputs, check_cache=True, load_train_num=-1,
                                          tokenize_batch_size=tokenize_batch_size)
        args.source_sequence_length = data_dict["x_train"]["source_input_ids"].shape[-1]
        args.target_sequence_length = data_dict["y_train"]["target_input_ids"].shape[-1]
        return data_dict
    else:
        # when more tasks are supported -> todo
        pass
