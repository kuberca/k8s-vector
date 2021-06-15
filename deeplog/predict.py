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
    hdfs = set()
    # hdfs = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

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

    # Hyperparameters
    num_classes = 250
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=30.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=5, type=int)
    parser.add_argument('-test_ok', default="../kl4-2.log.ok.template.all", type=str)
    parser.add_argument('-test_err', default="../kl4-2.log.err.template.all", type=str)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    # time_num_candidates = 3
    normal = args.test_ok
    abnormal = args.test_err

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate(normal)
    test_abnormal_loader = generate(abnormal)
    test_normal_loader = DataLoader(test_normal_loader, batch_size=1, shuffle=False, pin_memory=True)
    test_abnormal_loader = DataLoader(test_abnormal_loader, batch_size=1, shuffle=False, pin_memory=True)
    TP = 0
    FP = 0
    TP_TIME = 0
    FP_TIME = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for step, (seq, label, label_time) in enumerate(test_normal_loader):
            # seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            # label = torch.tensor(label).view(-1).to(device)
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            label = label.clone().detach().view(-1).to(device)
            label_time = label_time.clone().detach().view(-1).to(device)
            output, output_time = model(seq)

            # args = torch.argsort(output, 1)
            # args1 = args[0]

            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            predicted_time = torch.argsort(output_time, 1)[0][-1]
            # print(output)
            # print(args)
            # print(args1)
            # print(predicted)
            # print(label)
            
            # print("seq: {}".format(seq))
            print('normal, label/predicted: {}, {}, time/predicted: {},{}'.format(label, predicted, label_time, predicted_time))

            isbreak = False

            # for label, we treat abnormal when actual label is not in predicted ones
            if label not in predicted:
                FP += 1
                isbreak = True

            # for time, we treat abnormal when actual time is larger than predicted
            if label_time > predicted_time:
                FP_TIME += 1
                isbreak = True
            # if isbreak:
            #     break
    with torch.no_grad():
        for step, (seq, label, label_time) in enumerate(test_abnormal_loader):
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            label = label.clone().detach().view(-1).to(device)
            label_time = label_time.clone().detach().view(-1).to(device)
            output, output_time = model(seq)

            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            predicted_time = torch.argsort(output_time, 1)[0][-1]
            isbreak = False

            print('abnormal, label/predicted: {}, {}, time/predicted: {},{}'.format(label, predicted, label_time, predicted_time))

            if label not in predicted:
                TP += 1
                isbreak = True
            if label_time > predicted_time:
                TP_TIME += 1
                isbreak = True
            # if isbreak:
            #     break

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    TN = len(test_normal_loader) - FP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)

    print('true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(TP, FP, FN, TN, P, R, F1))

    FN_TIME = len(test_abnormal_loader) - TP_TIME
    TN_TIME = len(test_normal_loader) - FP_TIME
    P_TIME = 100 * TP_TIME / (TP_TIME + FP_TIME)
    R_TIME = 100 * TP_TIME / (TP_TIME + FN_TIME)
    F1_TIME = 2 * P_TIME * R_TIME / (P_TIME + R_TIME)

    
    
    print('time true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, true negative (TN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(TP_TIME, FP_TIME, FN_TIME, TN_TIME, P_TIME, R_TIME, F1_TIME))
    print('Finished Predicting')
