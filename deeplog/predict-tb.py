import torch
import torch.nn as nn
import time
import argparse
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cpu")


def generate_old(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    # hdfs = set()
    labels = []
    output_times = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            arr = [i.split(",") for i in ln.strip().split()]
            ids = [int(i[0])-1 for i in arr]
            times = [getTimeBucket(float(i[1])) for i in arr]

            labels.append(ids)
            output_times.append(times)
            
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(labels)))
    return labels, output_times

num_time_bucket = 10

def getTimeBucket(t):
    # <1, <5, <10, <30, <60, <120, <180, <300,<600,>600
    if t < 1:
        return 0
    elif t < 5:
        return 1
    elif t < 10:
        return 2
    elif t < 30:
        return 3
    elif t < 60:
        return 4
    elif t < 120:
        return 5
    elif t < 180:
        return 6
    elif t < 300:
        return 7
    elif t < 600:
        return 8
    else:
        return 9


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    output_times = []
    with open(name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            # line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            arr = [i.split(",") for i in line.strip().split()]
            ids = [int(i[0])-1 for i in arr]
            times = [float(i[1]) for i in arr]

            for i in range(len(ids) - window_size):
                inputs.append(ids[i:i + window_size])
                outputs.append(ids[i + window_size])
                output_times.append(getTimeBucket(times[i + window_size-1]))
                # outputs.append((ids[i + window_size], times[i + window_size]))
                # output_times.append(times[i + window_size])
    
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    # dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs), torch.tensor(output_times))
    # return inputs, outputs, output_times
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs), torch.tensor(output_times))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.fc_time = nn.Linear(hidden_size, num_time_bucket)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        outlstm, _ = self.lstm(x, (h0, c0))
        out = self.fc(outlstm[:, -1, :])
        out_time = self.fc_time(outlstm[:, -1, :])
        return out, out_time


if __name__ == '__main__':

    suspecious_cnt = 5
    suspecious_prob = 8.0
    # some of the templates predicted not matching we ignore, because we know those are not very useful logs
    ignored = ["getSandboxIDByPodUID", "Event occurred", "Generating pod status", "Patch status for pod", "HTTP verb", "SyncLoop", 
        "Removing volume from desired state", "Waiting for volumes to attach", "computePodActions", "Creating hosts mount for container pod",
        "Exec-Probe", "PLEG Write status", "Probe succeeded probeType", "Already ran container do nothing", "Added volume to desired state", "SetUpPod took time"]
    # Hyperparameters
    num_classes = 218
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=50.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=3, type=int)
    parser.add_argument('-test_ok', default="../test.ok", type=str)
    parser.add_argument('-test_err', default="../test.err", type=str)
    parser.add_argument('-drain_output', default="../drain.out", type=str)
    parser.add_argument('-sim_file', default="../templates.txt.vec.sim", type=str)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    time_num_candidates = 3
    normal = args.test_ok
    abnormal = args.test_err
    sim_file = args.sim_file
    
    # load drain output
    # id,size,template
    # id starts from 0
    drain = args.drain_output
    templates = {}
    with open(drain) as f:
        for line in f:
            arr = line.split(",")
            templates[arr[0]] = arr[2]

    # load similairy mapping of template ids
    sim_map = {}
    with open(sim_file) as f:
         for line in f:
            arr = line.split(";")
            sim_map[arr[0]] = arr[1]

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader, time_normal_loader = generate_old(normal)
    test_abnormal_loader, time_abnormal_loader = generate_old(abnormal)
    TP = 0
    FP = 0
    TP_TIME = 0
    FP_TIME = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        
        for idx in range(len(test_normal_loader)):
            line = test_normal_loader[idx]
            line_time = time_normal_loader[idx]

            print("new file")
            suspecious = 0

            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                label_time = line_time[i + window_size -1 ]

                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                label_time = torch.tensor(label_time).view(-1).to(device)
                # seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                # label = label.clone().detach().view(-1).to(device)
                # label_time = label_time.clone().detach().view(-1).to(device)
                output, output_time = model(seq)

                # args = torch.argsort(output, 1)
                # args1 = args[0]

                prob, predicted = torch.sort(output, 1)
                prob = prob[0][-num_candidates:]
                predicted = predicted[0][-num_candidates:]



                predicted_time = torch.argsort(output_time, 1)[0][-time_num_candidates:]
                # print(output)
                # print(args)
                # print(args1)
                # print(predicted)
                # print(label)
                
                # print("seq: {}".format(seq))
                
                isbreak = False
                print('normal, input: {}, label/predicted: {}, {}, prob: {}, sum: {}'.format(torch.flatten(seq), label, predicted, prob, torch.sum(prob)))

                # for label, we treat abnormal when actual label is not in predicted ones
                if label not in predicted:
                    tpl = templates[str(label.item())]
                    ptpl = templates[str(predicted[-1].item())]

                    ignore = False
                    for i in ignored:
                        if i in ptpl:
                            ignore=True
                            break
                    if ignore:
                        continue
                    
                    # use sim_map to see if can skip
                    label_s = sim_map[str(label.item())]
                    predicted_s = [sim_map[str(pid.item())] for pid in predicted]
                    if label_s in predicted_s:
                        print("continue with sim map")
                        continue

                    if prob[-1] < suspecious_prob:
                        suspecious += 1
                        if suspecious < suspecious_cnt:
                            print("FP suspicious", suspecious)
                            continue

                    inputs = [templates[str(int(ij))] for ij in torch.flatten(seq).tolist()]
                    print('FP, label: {}'.format(label.item()))
                    print("".join(inputs))
                    print('FP, label: {}, template: {}, predicted: {}'.format(label.item(), tpl, ptpl))

                    FP += 1
                    isbreak = True
                    

                # for time, we treat abnormal when actual time is larger than predicted
                # if label_time > predicted_time[-1]:
                #     FP_TIME += 1
                #     isbreak = True
                if isbreak:
                    break

    with torch.no_grad():
        for idx in range(len(test_abnormal_loader)):
            line = test_abnormal_loader[idx]
            line_time = time_abnormal_loader[idx]
            print("new file ", idx+1)
            suspecious = 0


            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                label_time = line_time[i + window_size -1 ]

                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                label_time = torch.tensor(label_time).view(-1).to(device)

                output, output_time = model(seq)

                prob, predicted = torch.sort(output, 1)
                prob = prob[0][-num_candidates:]
                predicted = predicted[0][-num_candidates:]

                predicted_time = torch.argsort(output_time, 1)[0][-time_num_candidates:]
                isbreak = False

                print('abnormal, input: {}, label/predicted: {}, {}, prob: {}, sum: {}'.format(torch.flatten(seq), label, predicted, prob, torch.sum(prob)))


                if label not in predicted:
                    tpl = templates[str(label.item())]
                    ptpl = templates[str(predicted[-1].item())]
                    ignore = False
                    for i in ignored:
                        if i in ptpl:
                            ignore=True
                            break
                    if ignore:
                        continue

                    # use sim_map to see if can skip
                    label_s = sim_map[str(label.item())]
                    predicted_s = [sim_map[str(pid.item())] for pid in predicted]
                    if label_s in predicted_s:
                        print("continue with sim map")
                        continue

                    if prob[-1] < suspecious_prob:
                        suspecious += 1
                        if suspecious < suspecious_cnt:
                            print("TP suspicious", suspecious)
                            continue

                    inputs = [templates[str(int(ij))] for ij in torch.flatten(seq).tolist()]
                    print('TP, label: {}'.format(label.item()))
                    print("".join(inputs))
                    print('TP, label: {}, template: {}, predicted: {}'.format(label.item(), tpl, ptpl))

                    TP += 1
                    isbreak = True
                    
                # if label_time > predicted_time[-1]:
                #     TP_TIME += 1
                #     isbreak = True
                if isbreak:
                    break

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    TN = len(test_normal_loader) - FP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print('normal = FP + TN: {}, abnormal = TP + FN: {}'.format(len(test_normal_loader), len(test_abnormal_loader)))

    print('true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(TP, FP, FN, TN, P, R, F1))

    # FN_TIME = len(test_abnormal_loader) - TP_TIME
    # TN_TIME = len(test_normal_loader) - FP_TIME

    # # print('time true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, true negative (TN): {},'.format(TP_TIME, FP_TIME, FN_TIME, TN_TIME))

    # P_TIME = 100 * TP_TIME / (TP_TIME + FP_TIME)
    # R_TIME = 100 * TP_TIME / (TP_TIME + FN_TIME)
    # F1_TIME = 2 * P_TIME * R_TIME / (P_TIME + R_TIME)
    
    # print('time true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(TP_TIME, FP_TIME, FN_TIME, TN_TIME, P_TIME, R_TIME, F1_TIME))
    # print('Finished Predicting')
