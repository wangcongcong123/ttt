import os, json, pickle

def read_groundtruth(gt_dir="subs/golden"):
    event_types = ['positive', 'negative', 'can_not_test', 'death', 'cure']
    output = {}
    for each_event in event_types:
        with open(os.path.join(gt_dir, each_event + "_sol.jsonl"), "r") as f:
            for line in f:
                ins = json.loads(line)
                if ins["id"] not in output:
                    output[ins["id"]] = {}
                for key, slot_gt_list in ins["golden_annotation"].items():
                    output[ins["id"]][key] = [e.lower() for e in slot_gt_list]
    return output

grounds = read_groundtruth()

def levenshtein(s1, s2):
    '''
    from: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
    :param s1: first string
    :param s2: second string
    :return:
    '''
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def convert_to_candidates(id, slot_type, target, id2condidates):
    candidates = id2condidates[id][slot_type]
    intersection = set(candidates).intersection(set(target))
    if len(intersection) == 0:
        transformed_targets = []
        for t in [", ".join(target)]:
            min = 1000000
            min_index = 0
            for idx, c in enumerate(candidates):
                distance = levenshtein(c, t) / len(c)
                if distance < min:
                    min = distance
                    min_index = idx
            transformed_targets.append(candidates[min_index])
    else:
        transformed_targets = target
    return transformed_targets


def readJsonl(path):
    output = {}
    with open(path, 'r') as f:
        for line in f:
            ins = json.loads(line)
            output[ins["id"]] = ins["predicted_annotation"]
    return output


def read_id2text(data_file="data/middle/test.json"):
    output = {}
    with open(data_file, 'r') as f:
        for line in f:
            ins = json.loads(line)
            if ins["id"] not in output:
                output[ins["id"]] = {}
            output[ins["id"]][ins["slot_type"]] = ins["context"].lower()
    return output


def transform():
    dir = "subs"
    id2condidates = pickle.load(open(os.path.join(dir, "testid2candidates.pkl"), "rb"))
    # the testid2candidates.pkl can also obtained by calling read_id2text()
    # read the README file on how to create data/middle/test.json (is unlabeled before the annotation release)
    from_runs = ["run-1", "run-2", "run-3"]
    to_runs = ["post-run-1", "post-run-2", "post-run-3"]
    event_types = ['positive', 'negative', 'can_not_test', 'death', 'cure']
    for from_run, to_run in zip(from_runs, to_runs):
        for each_event in event_types:
            if not os.path.isdir(os.path.join(dir, to_run)):
                os.makedirs(os.path.join(dir, to_run))
            with open(os.path.join(dir, to_run, each_event + ".jsonl"), "w+") as out:
                preds = readJsonl(os.path.join(dir, from_run, each_event + '.jsonl'))
                for id, slot_example_preds in preds.items():
                    transformed_pred_dict = {}
                    for slot_key, slot_pred in slot_example_preds.items():
                        transformed_target = convert_to_candidates(id, slot_key, slot_pred, id2condidates)
                        transformed_pred_dict[slot_key] = transformed_target
                    out.write(json.dumps({"id": id, "predicted_annotation": transformed_pred_dict}) + "\n")
    print("done")


if __name__ == '__main__':
    transform()
