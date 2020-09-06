import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nlp import load_dataset
import urllib.request
import os,json,sys
import shutil
import tarfile

# these have been tested and work fine. more can be added to this list to test
MODELS_SUPPORT = ["distilbert-base-cased","bert-base-uncased", "bert-large-uncased", "google/electra-base-discriminator",
                  "google/electra-large-discriminator", "albert-base-v2", "roberta-base",
                  "t5-small","t5-base", "t5-large"]
# if using t5 models, the tasks has to be t2t* ones
TASKS_SUPPORT = ["single-label-cls", "t2t","translation"]
# in the future, more schedulers will be added, such as warmupconstant, warmupcosine, etc.
LR_SCHEDULER_SUPPORT = ["warmuplinear", "warmupconstant", "constant"]

ADDITIONAL_DATA_SUPPORT = {"wmt_en_ro":"https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro.tar.gz"}


def ts2jsonl(data_path,t5_prefix_source=True):
    """
    t5_prefix_source: by default, t5_prefix_source, "translate English to Romanian: " is prepended to source sequence as in: https://arxiv.org/pdf/1910.10683.pdf
    """
    sets=["train","val","test"]
    for set_name in sets:
        if os.path.isfile(os.path.join(data_path,set_name + ".source")):
            with open(os.path.join(data_path,set_name + ".source"), encoding="utf-8") as src:
                src_lines = src.readlines()
            with open(os.path.join(data_path,set_name + ".target"), encoding="utf-8") as src:
                tgt_lines = src.readlines()
            assert len(src_lines) == len(tgt_lines)
            if set_name == "train":
                with open(os.path.join(data_path,set_name + "_fixture.json"), "w+", encoding="utf-8") as f:
                    for source, target in tqdm(zip(src_lines[:10000], tgt_lines[:10000])):
                        f.write(json.dumps({"source": "translate English to Romanian: "+source.strip(), "target": target.strip()}) + "\n")

            with open(os.path.join(data_path,set_name + ".json"), "w+", encoding="utf-8") as f:
                for source, target in tqdm(zip(src_lines, tgt_lines),desc=f"converting original data: {data_path} to jsonl formats"):
                    if t5_prefix_source:
                        source="translate English to Romanian: "+ source
                    f.write(json.dumps({"source": source.strip(), "target": target.strip()}) + "\n")



def data_check(args, sample_val_from_train=True, val_sample_portion=0.1):

    if not os.path.isfile(os.path.join(args.data_path,"train.json")):
        # if it is from: https://huggingface.co/nlp/viewer, load it
        # if there is no validation set, by default, here a random sampling from the train (0.1 ratio) is used to form the val set
        # so far it works well for single-sequence cls datasets such as "glue/sst2", "20newsgroup", "ag_news", "imdb", "sentiment140", etc.
        # support more all kinds of other datasets that are availale in nlp, such as sequence-pair cls datasets (NLI), qa datasets, etc. -> todo
        set_name = args.data_path.split("/")[1:]
        try:
            dataset = load_dataset(*set_name)
            target_examples_dict = {}
            assert "train" in dataset, "not found train set in the given nlp dataset, make sure you give the correct name as listed in: https://huggingface.co/nlp/viewer/"
            label_names = dataset["train"].features["label"].names
            for set_key, examples in tqdm(dataset.items(),
                                          desc=f"not found the data locally, try to load {set_name} from nlp lib."):
                if set_key == "validation":
                    set_key = "val"
                if set_key not in target_examples_dict:
                    target_examples_dict[set_key] = []

                if set_key == "test" and "sst2" in set_name:
                    # sst2 does not have ground truths in test set from the nlp lib.
                    # here a special branch is processed here:
                    # to download a sst-test set with the same format as the train and val set loaded here
                    # from here: https://ucdcs-student.ucd.ie/~cwang/sst2-test.json
                    # for model testing
                    sst2_test = urllib.request.urlopen("https://ucdcs-student.ucd.ie/~cwang/sst2-test.json")
                    for line in sst2_test:
                        decoded_line = line.decode("utf-8").strip()
                        target_examples_dict[set_key].append(decoded_line)
                else:
                    for example in examples:
                        example["label"] = label_names[example["label"]]  # convert to raw label
                        if "sentence" in example:
                            example["text"] = example["sentence"]
                            del example["sentence"]
                        target_examples_dict[set_key].append(json.dumps(example))

            if sample_val_from_train and "val" not in target_examples_dict:
                train, val = train_test_split(target_examples_dict["train"], test_size=val_sample_portion,
                                              random_state=42)
                target_examples_dict["train"] = train
                target_examples_dict["val"] = val

            to_folder = os.path.join("data", *set_name)

            if not os.path.isdir(to_folder):
                os.makedirs(to_folder)

            splits_dict = {}
            for set_key, examples in tqdm(target_examples_dict.items(), desc="writing..."):
                with open(os.path.join(to_folder, set_key + ".json"), "w+") as target:
                    target.write("\n".join(examples))
                    splits_dict[set_key] = len(examples)
            with open(os.path.join(to_folder, "splits.txt"), "w+") as tgt:
                tgt.write(json.dumps(splits_dict, indent=2))

        except:
            if set_name[0] in ADDITIONAL_DATA_SUPPORT.keys():
                fstream = urllib.request.urlopen(ADDITIONAL_DATA_SUPPORT[set_name[0]])
                tfile = tarfile.open(fileobj=fstream, mode="r|gz")
                tfile.extractall(path="data")
                if set_name[0] == "wmt_en_ro":
                    ts2jsonl(args.data_path)
            else:
                print("data not found")
                sys.exit(0)


def sanity_check(args):
    # auto-construct some args
    # check if data exists
    data_check(args)

    output_folder = args.model_select + "_" + args.task + "_" + "-".join(args.data_path.split("/")[1:])
    output_path = os.path.join("tmp", output_folder)
    args.output_folder = output_folder
    args.output_path = output_path

    if args.do_train:
        if os.path.isdir(output_path):
            out = input(
                "Output directory ({}) already exists and is not empty, you wanna remove it before start training? (y/n)".format(
                    output_path))
            if out.lower() == "y":
                shutil.rmtree(output_path)
                os.makedirs(output_path, exist_ok=True)
            else:
                sys.exit(0)
        else:
            os.makedirs(output_path, exist_ok=True)

    if "t5" in args.model_select:
        assert "t2t" in args.task or "translation" in args.task, f"t5 models does not support when --task == {args.task}"

    assert args.model_select in MODELS_SUPPORT, F"set --model_select to be in {MODELS_SUPPORT}"
    assert args.task in TASKS_SUPPORT, F"set --task to be in {TASKS_SUPPORT}"
    assert args.scheduler in LR_SCHEDULER_SUPPORT, F"set --scheduler to be in {TASKS_SUPPORT}"
    if "t5" in args.model_select:
        assert "t2t" in args.task or "translation" in args.task,"t5 models (--model_select) only support t2t and translation tasks (--task)"

    if "t5" not in args.model_select:
        assert "t2t" not in args.task, "BERT-like models (--model_select) only support non t2t tasks (--task)"

    if "translation" in args.task:
        assert "t5" in args.model_select, "translation task now is only compatible with T5 models"


class Args:
    '''
    a Args class that maintain the same default args as argparse.ArgumentParser
    '''
    model_select="bert-base-uncased"
    data_path="data/glue/sst2"
    task="single-label-cls"
    per_device_train_batch_size=8
    eval_batch_size=32
    num_epochs_train=6
    log_steps=400
    max_seq_length=128
    max_src_length=128
    max_tgt_length=20
    lr=5e-5
    warmup_ratio=0.1
    patience=20
    scheduler="warmuplinear"
    seed=122
    eval_on="acc"
    keep_ck_num=3
    ck_index_select=0
    do_train=False
    do_eval=False
    do_test=False
    use_gpu=False
    use_tpu=False
    use_tb=False
    tpu_address="x.x.x.x"
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_args():
    parser = argparse.ArgumentParser(description='Hyper params')

    parser.add_argument('--model_select', type=str, default="bert-base-uncased",
                        help='model select from MODEL_MAP')

    parser.add_argument('--data_path', type=str, default="data/glue/sst2",
                        help='data path')

    parser.add_argument('--task', type=str, default="single-label-cls",
                        help='tasks available in TASKS_SUPPORT')

    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='input batch size for training')

    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='input batch size for training')

    parser.add_argument('--num_epochs_train', type=int, default=6,
                        help='number of epochs to train')

    parser.add_argument('--log_steps', type=int, default=400,
                        help='logging steps for evaluation based on global step if it is not -1 and based on epoch if it is -1, and tracking metrics using tensorboard if use_tb is active')

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='maximum sequence length of samples in a batch for training')

    parser.add_argument('--max_src_length', type=int, default=128,
                        help='only working for t5-like t2t-based tasks, maximum source sequence length of samples in a batch for training')

    parser.add_argument('--max_tgt_length', type=int, default=20,
                        help='only working for t5-like t2t-based tasks, maximum target sequence length of samples in a batch for training')

    parser.add_argument('--source_field_name', type=str, default="text",
                        help='only working for t5-like t2t-based tasks, the source field name of the provided jsonl-formatted data')

    parser.add_argument('--target_field_name', type=str, default="label",
                        help='only working for t5-like t2t-based tasks, the target field name of the provided jsonl-formatted data')


    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')

    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup_ratio, only working if scheduler is not constant')

    parser.add_argument('--patience', type=int, default=20,
                        help='patience based on the log_steps')

    parser.add_argument('--scheduler', type=str, default="warmuplinear",
                        help='scheduler')

    parser.add_argument('--seed', type=int, default=122,
                        help='random seed')

    parser.add_argument('--eval_on', type=str, default="acc",
                        help='eval on for best ck saving and patience-based training early stop')

    parser.add_argument('--keep_ck_num', type=int, default=3,
                        help='keep_ck_num except for the best ck (evaluated on validation set using the metric specified by --eval_on')

    parser.add_argument('--ck_index_select', type=int, default=0,
                        help='ck_index_select, use the best one by default, negative one to specify a latest one, working when --do_test is active')

    parser.add_argument(
        "--do_train", action="store_true", help="Do training"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="do evaluation on validation set for saving checkpoint"
    )
    parser.add_argument(
        "--do_test", action="store_true", help="eval on test set if test set is availale"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="use gpu?"
    )

    parser.add_argument(
        "--use_tpu", action="store_true", help="use tpu? "
    )
    parser.add_argument(
        "--use_tb", action="store_true", help="use tensorboard for tracking training process, default save to ./runs"
    )

    parser.add_argument('--tpu_address', type=str, default="x.x.x.x",
                        help='cloud tpu address if using tpu')

    parser.add_argument(
        "--default_store", action="store_true",
        help="Store datasets, weights, logs, and relevant details to folders by default?"
    )

    args = parser.parse_args()
    return args