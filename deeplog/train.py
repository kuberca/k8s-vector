import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                output_times.append([times[i + window_size-1]])
                # outputs.append((ids[i + window_size], times[i + window_size]))
                # output_times.append(times[i + window_size])
    
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs), torch.tensor(output_times, dtype=torch.float))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.fc_time = nn.Linear(hidden_size, 1)

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
    num_epochs = 30
    batch_size = 2048
    # input_size = 1
    input_size = 1              # (logid, timediff)
    model_dir = 'model'
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-template_file', default="/Users/junzhou/code/rca/data/pk8s/kubelet/alllog/kl3-2.log.ok.template.all", type=str)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    seq_dataset = generate(args.template_file)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label, label_times) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output, output_time = model(seq)
            # loss1 = criterion(output, label).to(device)
            loss2 = criterion_time(output_time, label_times).to(device)
            # loss = loss1 + loss2
            loss = loss2
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
