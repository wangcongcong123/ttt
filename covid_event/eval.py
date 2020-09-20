'''
copyright declaration:
this eval script is adapted from: https://github.com/viczong/extract_COVID19_events_from_Twitter/tree/master/shared_task
'''
import argparse
import json
import numpy as np

### read file
def readJSONLine(path):
    output = []
    with open(path, 'r') as f:
        for line in f:
            output.append(json.loads(line))
    return output

### evaluation script
def runEvaluation(system_predictions, golden_predictions):
    ## read in files
    golden_predictions_dict = {}
    for each_line in golden_predictions:
        golden_predictions_dict[each_line['id']] = each_line

    ## question tags
    question_tag = [i for i in golden_predictions[0]['golden_annotation'] if 'part2' in i]
    ## evaluation
    result = {}
    for each_task in question_tag:
        # evaluate curr task
        curr_task = {}
        TP, FP, FN = 0.0, 0.0, 0.0
        for each_line in system_predictions:
            curr_sys_pred = [i.lower() for i in each_line['predicted_annotation'][each_task] if \
                             i != 'Not Specified' and i != 'not specified' and i != 'not_effective']
            #             print(golden_predictions_dict[each_line['id']]['golden_annotation'][each_task])
            curr_golden_ann = [i.lower() for i in
                               golden_predictions_dict[each_line['id']]['golden_annotation'][each_task] \
                               if i != 'Not Specified' and i != 'not specified' and i != 'not_effective']
            #             print(curr_sys_pred, curr_golden_ann)
            if len(curr_golden_ann) > 0:
                for predicted_chunk in curr_sys_pred:
                    if predicted_chunk in curr_golden_ann:
                        TP += 1  # True positives are predicted spans that appear in the gold labels.
                    else:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.
                for gold_chunk in curr_golden_ann:
                    if gold_chunk not in curr_sys_pred:
                        FN += 1  # False negatives are gold spans that weren't in the set of spans predicted by the model.
            else:
                if len(curr_sys_pred) > 0:
                    for predicted_chunk in curr_sys_pred:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.
        # print
        if TP + FP == 0:
            P = 0.0
        else:
            P = TP / (TP + FP)

        if TP + FN == 0:
            R = 0.0
        else:
            R = TP / (TP + FN)

        if P + R == 0:
            F1 = 0.0
        else:
            F1 = 2.0 * P * R / (P + R)

        curr_task["F1"] = F1
        curr_task["P"] = P
        curr_task["R"] = R
        curr_task["TP"] = TP
        curr_task["FP"] = FP
        curr_task["FN"] = FN
        N = TP + FN
        curr_task["N"] = N
        # print(curr_task)
        result[each_task.replace('.Response', '')] = curr_task
        # print
    #         print(each_task.replace('.Response', ''))
    #         print('P:', curr_task['P'], 'R:', curr_task['R'], 'F1:', curr_task['F1'])
    #         print('=======')
    ### calculate micro-F1
    all_TP = np.sum([i[1]['TP'] for i in result.items()])
    all_FP = np.sum([i[1]['FP'] for i in result.items()])
    all_FN = np.sum([i[1]['FN'] for i in result.items()])

    all_P = all_TP / (all_TP + all_FP)
    all_R = all_TP / (all_TP + all_FN)
    all_F1 = 2.0 * all_P * all_R / (all_P + all_R)

    ## append
    result['micro'] = {}
    result['micro']['TP'] = all_TP
    result['micro']['FP'] = all_FP
    result['micro']['FN'] = all_FN
    result['micro']['P'] = all_P
    result['micro']['R'] = all_R
    result['micro']['F1'] = all_F1
    result['micro']['N'] = all_TP + all_FN
    #     print('micro F1', all_F1)
    return result

if __name__ == '__main__':
    ##### Attention: replace YOUR_TEAM_NAME with your actual team name
    ## YOUR_TEAM_NAME = 'OSU_NLP'
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--run_name', type=str, default="run-2",
                        help='run name for evaluation on test set (2500 tweets, 500 per event)')
    args = parser.parse_args()
    input_path = './subs/' + args.run_name +'/'
    golden_path = './subs/golden/'
    team_name = input_path.split('/')[-2]
    print('team name:', team_name)
    ### score each category
    category_flag = ['positive', 'negative', 'can_not_test', 'death', 'cure']
    curr_team = {}
    curr_team['team_name'] = team_name
    ## loop each category
    all_category_results = {}
    for each_category in category_flag:
        ## read in data
        curr_pred = readJSONLine(input_path   + each_category + '.jsonl')
        curr_sol = readJSONLine(golden_path + each_category + '_sol.jsonl')
        ## generate result
        curr_result = runEvaluation(curr_pred, curr_sol)
        ## print
        print(team_name, each_category, 'F1:', curr_result['micro']['F1'])
        ## append result
        all_category_results[each_category] = curr_result
    ### overall
    all_cate_TP = np.sum([i[1]['micro']['TP'] for i in all_category_results.items()])
    all_cate_FP = np.sum([i[1]['micro']['FP'] for i in all_category_results.items()])
    all_cate_FN = np.sum([i[1]['micro']['FN'] for i in all_category_results.items()])
    # print(all_cate_TP + all_cate_FN)
    ### micro-F1
    all_cate_P = all_cate_TP / (all_cate_TP + all_cate_FP)
    all_cate_R = all_cate_TP / (all_cate_TP + all_cate_FN)
    all_cate_F1 = 2.0 * all_cate_P * all_cate_R / (all_cate_P + all_cate_R)
    curr_team['category_perf'] = all_category_results
    merged_performance = {}
    merged_performance['TP'] = all_cate_TP
    merged_performance['FP'] = all_cate_FP
    merged_performance['FN'] = all_cate_FN
    merged_performance['P'] = all_cate_P
    merged_performance['R'] = all_cate_R
    merged_performance['F1'] = all_cate_F1
    curr_team['overall_perf'] = merged_performance
    print('-----')
    print(merged_performance)
    print(team_name, 'overall', 'F1:', all_cate_F1)
    print('======')
