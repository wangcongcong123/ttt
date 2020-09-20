import copy
import json
import os

from tqdm import tqdm

eventname2question = {
    "positive": "Does this tweet report an individual or a small group of people who is tested postive for coronavirus?",
    "negative": "Does this tweet report an individual or a small group of people who is tested negative for coronavirus?",
    "cure_and_prevention": "Does this tweet report cure and prevention for coronavirus?",
    "can_not_test": "Does this tweet report an individual or a small group of people who can not be tested for coronavirus?",
    "death": "Does this tweet report dealth for coronavirus?",
}

short2question_positive = {"name": "Who is tested positive?",
                           "close_contact": "Who is in close contact with the person tested positive?",
                           "employer": "Who is the employer of the people tested positive?",
                           "recent_travel": "Where did the people tested positive recently visit?",
                           "relation": "Does the infected person have a relationship with the author of the tweet?",
                           "gender": "What is the gender of the people tested positive?",
                           "age": "What is the age of the people tested positive?",
                           "when": "When are tested positive cases reported?",
                           "where": "Where are tested positive cases reported?", }

short2choices_positive = {"name": ["chunks", "author of the tweet", "not specified"],
                          "close_contact": ["chunks", "author of the tweet", "not specified"],
                          "employer": ["chunks", "not specified"],
                          "recent_travel": ["chunks", "near author of the tweet", "not specified"],
                          "relation": ["yes", "no", "not specified"],
                          "gender": ["male", "female", "not specified"],
                          "age": ["chunks", "not specified"],
                          "when": ["chunks", "not specified"],
                          "where": ["chunks", "near author of the tweet", "not specified"], }

short2question_negative = {"name": "Who is tested negative?",
                           "close_contact": "Who is in close contact with the person tested negative?",
                           "relation": "Does the infected person have a relationship with the author of the tweet?",
                           "gender": "What is the gender of the people tested negative?",
                           "age": "What is the age of the people tested negative?",
                           "when": "When are tested negative cases reported?",
                           "where": "Where are tested negative cases reported?",
                           "how_long": "How long does it take to get to know the test results?"}

short2choices_negative = {"name": ["chunks", "author of the tweet", "not specified"],
                          "close_contact": ["chunks", "author of the tweet", "not specified"],
                          "relation": ["yes", "no", "not specified"],
                          "gender": ["male", "female", "not specified"],
                          "age": ["chunks", "not specified"],
                          "when": ["chunks", "not specified"],
                          "where": ["chunks", "near author of the tweet", "not specified"],
                          "how_long": ["chunks", "not specified"], }

short2question_can_not_test = {"name": "Who can not get a test?",
                               "symptoms": "Is the untested person currently experiencing any COVID-19 related symptoms?",
                               "relation": "Does the untested person have a relationship with the author of the tweet?",
                               "when": "When is the can’t-be-tested situation reported?",
                               "where": "Where is the can’t-be-tested situation reported?", }

short2choices_can_not_test = {"name": ["chunks", "author of the tweet", "not specified"],
                              "symptoms": ["yes", "no", "not specified"],
                              "relation": ["yes", "no", "not specified"],
                              "when": ["chunks", "not specified"],
                              "where": ["chunks", "near author of the tweet", "not specified"], }

short2question_dealth = {"name": "Who is dead for coronavirus?",
                         "symptoms": "Did the person who was dead experience COVID-19 related symptoms?",
                         "relation": "Does the deceased person have a relationship with the author of the tweet?",
                         "when": "When is the dead case reported?",
                         "where": "Where is the dead case reported?",
                         "age": "What is the age of the people who is dead of COVID-19?", }

short2choices_death = {"name": ["chunks", "author of the tweet", "not specified"],
                       "symptoms": ["yes", "no", "not specified"],
                       "relation": ["yes", "no", "not specified"],
                       "when": ["chunks", "not specified"],
                       "where": ["chunks", "near author of the tweet", "not specified"],
                       "age": ["chunks", "not specified"], }

short2question_cure_and_prevention = {"opinion": "Does the author of tweet believe the cure method is effective?",
                                      "what_cure": "What is the cure for coronavirus mentioned by the author of the tweet?",
                                      "who_cure": "Who is promoting the cure for coronavirus?", }

short2choices_cure_and_prevention = {"opinion": ["not_effective", "effective"],
                                     "what_cure": ["chunks", "not specified"],
                                     "who_cure": ["chunks", "author of the tweet", "not specified"], }

eventname2questionmapping = {"positive": short2question_positive,
                             "negative": short2question_negative,
                             "can_not_test": short2question_can_not_test,
                             "death": short2question_dealth,
                             "cure_and_prevention": short2question_cure_and_prevention,
                             }

eventname2choicesnmapping = {"positive": short2choices_positive,
                             "negative": short2choices_negative,
                             "can_not_test": short2choices_can_not_test,
                             "death": short2choices_death,
                             "cure_and_prevention": short2choices_cure_and_prevention, }

event_type2part2annotation_keys = {
    "can_not_test": [
        "part2-relation.Response",
        "part2-symptoms.Response",
        "part2-name.Response",
        "part2-when.Response",
        "part2-where.Response"
    ],
    "cure_and_prevention": [
        "part2-opinion.Response",
        "part2-what_cure.Response",
        "part2-who_cure.Response"
    ],
    "negative": [
        "part2-age.Response",
        "part2-close_contact.Response",
        "part2-gender.Response",
        "part2-how_long.Response",
        "part2-name.Response",
        "part2-relation.Response",
        "part2-when.Response",
        "part2-where.Response"
    ],
    "death": [
        "part2-age.Response",
        "part2-name.Response",
        "part2-relation.Response",
        "part2-symptoms.Response",
        "part2-when.Response",
        "part2-where.Response"
    ],
    "positive": [
        "part2-age.Response",
        "part2-close_contact.Response",
        "part2-employer.Response",
        "part2-gender.Response",
        "part2-name.Response",
        "part2-recent_travel.Response",
        "part2-relation.Response",
        "part2-when.Response",
        "part2-where.Response"
    ]
}

def get_part1_new_example(example):
    event_type = example["event_type"]
    new_example = {}
    new_example["id"] = example["id_str"]  # id
    new_example["event_type"] = event_type  # event_type
    new_example["slot_type"] = "part1.Response"  # slot_type
    new_example["context"] = example["full_text"]  # context
    new_example["question"] = eventname2question[event_type]  # question
    new_example["choices"] = "yes, no"  # candidate choices
    new_example["answer"] = example["annotation"]["part1.Response"][0]
    return new_example

def get_text_chunks(example):
    full_text = example["full_text"]  # context
    candidate_chunks_offsets = example["candidate_chunks_offsets"]
    text_chunks = [full_text[each[0]:each[1]] for each in candidate_chunks_offsets]
    return text_chunks


def get_total(filepath):
    total = 0
    with open(filepath, "r") as f:
        for line in f:
            total += 1
    return total


def build_test(filepath, event_type="can_not_test"):
    new_examples = []
    with open(filepath, "r") as f:
        for line in tqdm(f):
            example = json.loads(line.strip())
            text_chunks = get_text_chunks(example)
            part2annotation_keys = event_type2part2annotation_keys[event_type]
            for each_part2_annotation_key in part2annotation_keys:
                slot_key = each_part2_annotation_key.split(".")[0].split("-")[-1]
                slot_question = eventname2questionmapping[event_type][slot_key]
                slot_candidate_choices = copy.deepcopy(eventname2choicesnmapping[event_type][slot_key])
                if "chunks" in slot_candidate_choices:
                    slot_candidate_choices.remove("chunks")
                    slot_candidate_choices.extend(text_chunks)
                new_example = {}
                full_text = example["text"]
                new_example["id"] = example["id"]  # id
                new_example["event_type"] = event_type  # event_type
                new_example["slot_type"] = each_part2_annotation_key  # slot_type
                new_example["context"] = full_text  # context
                new_example["question"] = slot_question  # question
                new_example["candidates"] = slot_candidate_choices  # candidate choices
                new_example["choices"] = ", ".join(slot_candidate_choices)  # candidate choices
                new_examples.append(new_example)
    return new_examples


def build_train_val(filepath, test_size=0.1, out_folder="data/middle"):
    total = get_total(filepath)
    test_no = int(total * test_size)
    # new_examples = []
    train, val = [], []
    NO_CONSENSUS_count = 0
    with open(filepath, "r") as f:
        for line in f:
            if total <= test_no:
                new_examples = val
            else:
                new_examples = train
            example = json.loads(line.strip())
            if example["annotation"]["part1.Response"][0] == "yes":
                new_example = get_part1_new_example(example)
                new_examples.append(new_example)
                text_chunks = get_text_chunks(example)
                event_type = example["event_type"]
                annotation = example["annotation"]
                annotation.pop("part1.Response")
                for each_part2_annotation_key, value in annotation.items():
                    if value == "NO_CONSENSUS":
                        NO_CONSENSUS_count += 1
                        continue
                    slot_key = each_part2_annotation_key.split(".")[0].split("-")[-1]
                    slot_question = eventname2questionmapping[event_type][slot_key]
                    slot_candidate_choices = copy.deepcopy(eventname2choicesnmapping[event_type][slot_key])
                    if "chunks" in slot_candidate_choices:
                        slot_candidate_choices.remove("chunks")
                        slot_candidate_choices.extend(text_chunks)
                    new_example = {}
                    full_text = example["full_text"]
                    new_example["id"] = example["id_str"]  # id
                    new_example["event_type"] = event_type  # event_type
                    new_example["slot_type"] = each_part2_annotation_key  # slot_type
                    new_example["context"] = full_text  # context
                    new_example["question"] = slot_question  # question
                    new_example["choices"] = ", ".join(slot_candidate_choices)  # candidate choices
                    answers = []
                    for v in value:
                        if isinstance(v, str):
                            if v.lower() in ["no_cure", "not_effective", "no_opinion"]:
                                answers.append("not_effective")
                            else:
                                answers.append(v.lower())
                        else:
                            answers.append(full_text[v[0]:v[1]])
                    assert set(answers).intersection(set(slot_candidate_choices)) != set()
                    new_example["answer"] = ", ".join(answers)  # answer
                    new_examples.append(new_example)
            else:
                new_example = get_part1_new_example(example)
                new_examples.append(new_example)
            total -= 1
    print(f"there are {NO_CONSENSUS_count} NO_CONSENSUS")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    with open(out_folder + "/train.json", "w+") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(out_folder + "/val.json", "w+") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")
    print(f"done building train and val from {filepath}, written to {out_folder}")


def construct(data_path, use_choices=True, no_answer=False, out_folder="data/final"):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    with(open(out_folder + "/" + data_path.split("/")[-1], "w+")) as tgt:
        with open(data_path, "r") as f:
            for line in tqdm(f, desc="reading..."):
                example = json.loads(line.strip())
                source_sequence = "context: " + example["context"] + " question: " + example["question"]
                if use_choices:
                    source_sequence += " choices: " + example["choices"]
                if not no_answer:
                    target_sequence = example["answer"]
                    tgt.write(json.dumps(
                        {"id": example["id"], "event_type": example["event_type"], "slot_type": example["slot_type"],
                         "source": source_sequence, "target": target_sequence}) + "\n")
                else:
                    tgt.write(json.dumps(
                        {"id": example["id"], "event_type": example["event_type"], "slot_type": example["slot_type"],
                         "source": source_sequence, "candidates": example["candidates"]}) + "\n")
        print("done source and target sequences construction")

if __name__ == '__main__':
    '''
    sequence construction examples:
        X1: context: @CPRewritten I heard from @Corona_Bot__ that G is tested positive of COVID-19. question: Who is tested positive? choices: Not Specified, A, B, C.
        y1: A
        X2: context: @CPRewritten I heard from @Corona_Bot__ that G is tested positive of COVID-19. question: Does this message report an individual or a small group of people who is tested postive for coronavirus. choices: yes, no.
        y2: yes
    '''
    build_train_val("data/corpus.json", test_size=0.1, out_folder="data/middle")
    construct("data/middle/train.json", out_folder="data/final")
    construct("data/middle/val.json", out_folder="data/final")

    ### when test set is available, uncomment the following
    # can_not_test_examples = build_test("shared_task-test-can_not_test.jsonl", event_type="can_not_test")
    # cure_and_prevention_examples = build_test("shared_task-test-cure.jsonl", event_type="cure_and_prevention")
    # death_examples = build_test("shared_task-test-death.jsonl", event_type="death")
    # negative_examples = build_test("shared_task-test-negative.jsonl", event_type="negative")
    # postive_examples = build_test("shared_task-test-positive.jsonl", event_type="positive")
    # all = can_not_test_examples + cure_and_prevention_examples + death_examples + negative_examples + postive_examples
    # with open("../test.json", "w+") as f:
    #     for ex in all:
    #         f.write(json.dumps(ex) + "\n")
    # construct("ori/test.json", no_answer=True)
