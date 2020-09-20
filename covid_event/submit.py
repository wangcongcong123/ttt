from tqdm import tqdm
import json, os

def process_target(event_type, slot_type, target):
    # For opinion slot in cure & prevention category, you shall only have two labels: "effective" and "not_effective" ("no_opinion", "no_cure" and "not_effective" will be merged into "not_effective")
    if event_type == "cure_and_prevention" and "opinion" in slot_type:
        if "no_opinion" in target:
            target.remove("no_opinion")
            target.append("not_effective")
        if "no_cure" in target:
            target.remove("no_cure")
            target.append("not_effective")
    # "I" (along with its variations) shall be replaced with "AUTHOR OF THE TWEET" both for NAME and CLOSE_CONTACT slot for all categories.
    if "name" in slot_type or "close_contact" in slot_type:
        if "i" in target:
            # It doesn't matter if your predictions are lowercased or uppercased.
            target.remove("i")
            target.append("author of the tweet")
        if "i'm" in target:
            target.remove("i'm")
            target.append("author of the tweet")
    # For relation slot and symptoms slot, you shall only have two labels: "yes" and "not specified" ("no" and "not specified" will be merged into "not specified").
    if "relation" in slot_type or "symptoms" in slot_type:
        if "no" in target:
            target.remove("no")
            target.append("not specified")
    return list(set(target))  # do not submit repeated answers

def convert(data_path, output_path="t5_sub", annotation_key="predicted_annotation"):
    with open(data_path, "r") as f:
        event2converted_examples = {}
        converted_examples = []
        converted_one_example = {'id': '', annotation_key: {}}
        last_event = ""
        last_id = ""
        switch = False

        for line in tqdm(f):
            example = json.loads(line.strip())
            slot_type = example["slot_type"]
            if "part2" in slot_type:
                event_type = example["event_type"]
                if event_type != last_event:
                    if last_event != "":
                        switch = False
                        converted_examples.append(converted_one_example)
                        event2converted_examples[last_event] = converted_examples
                    converted_examples = []

                id = example["id"]
                target = example["target"].split(", ")

                if id != last_id:
                    if last_id != "" and switch:
                        converted_examples.append(converted_one_example)

                    # Death: "symptoms" slot will be excluded
                    # Tested Negative: "how long" slots will be excluded
                    if (event_type == "death" and "symptoms" in slot_type) or (
                            event_type == "negative" and "how_long" in slot_type):
                        continue

                    target = process_target(event_type, slot_type, target)

                    converted_one_example = {'id': id,
                                             annotation_key: {slot_type: target}}

                else:
                    switch = True
                    # Death: "symptoms" slot will be excluded
                    # Tested Negative: "how long" slots will be excluded
                    if (event_type == "death" and "symptoms" in slot_type) or (
                            event_type == "negative" and "how_long" in slot_type):
                        continue
                    target = process_target(event_type, slot_type, target)
                    converted_one_example[annotation_key][slot_type] = target

                last_id = id
                last_event = event_type
        converted_examples.append(converted_one_example)
        event2converted_examples[last_event] = converted_examples

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    for event, examples in event2converted_examples.items():
        if "cure" in event:
            event = "cure"
        output_file_path = output_path + event + ".jsonl"
        with open(output_file_path, "w+") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
            print(f'done writing to {output_file_path}')
    return event2converted_examples

if __name__ == '__main__':
    convert("preds/val_preds.json", output_path="subs/val-run-1/")
