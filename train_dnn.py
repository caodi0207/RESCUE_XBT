import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Stock Value Prediction Model')
parser.add_argument('--data', type=str, default='./data/sz002821',
                    help='location of the data')
parser.add_argument('--nhid', type=int, default=50,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--lr_decay', type=float, default=0.25,
                    help='decay lr by the rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--use_define', action='store_true')
parser.add_argument('--save', type=str,  default='./models/dnn.pt',
                    help='path to save the final model')
args = parser.parse_args()

print '{:=^30}'.format('all args')
for arg in vars(args):
    print ' '.join(map(str, (arg, getattr(args, arg))))

class DataIter(object):
    def __init__(self, path, batch_size, cuda=False):
        self.path = path
        self.batch_size = batch_size
        self.cuda = cuda

        self.build_data()
        self.batchify()

    def build_data(self):
        data_type = np.dtype([('features', 'f8', (20, )), ('labels1', 'i8', (1, )), ('labels2', 'i8', (1, ))])
        data = np.loadtxt(self.path, data_type, delimiter=' ')
        features = data['features']
        labels1 = data['labels1']
        labels2 = data['labels2']
        for i in range(labels1.shape[0]):
            labels1[i] += 100
        for i in range(labels2.shape[0]):
            labels2[i] += 100
        for i in range(labels1.shape[0]):
            labels1[i]  = labels1[i] *201 + labels2[i]
        features = torch.Tensor(data['features'])
        labels1 = torch.LongTensor(labels1)
        labels2 = torch.LongTensor(labels2)

        self.data = features
        self.label = labels1
        return

    def batchify(self):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = self.data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.data[:nbatch * self.batch_size]
        label = self.label.t().contiguous()
        label = label[:, :nbatch * self.batch_size]
        # Evenly divide the data across the bsz batches.
        data = data.view(-1, self.batch_size, 20).contiguous()
        label = label.view(-1, self.batch_size).contiguous()
        self.data = data.cuda() if self.cuda else data
        self.label = label.cuda() if self.cuda else label

    def __iter__(self):
        for idx in range(0, self.data.size(0) - 1, 1):
            data = Variable(self.data[idx])
            target = Variable(self.label[idx])
            yield data, target

    def __len__(self):
        return self.data.size(0)



class DNNModel(nn.Module):
    def __init__(self, nfed, nhid, noutputs, dropout=0.5):
        super(DNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Linear(nfed, nhid)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(nhid, noutputs)

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.fill_(0)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.encoder(input)
        hidden = self.sigmoid(emb)
        output = self.drop(self.decoder(hidden))
        return output

def countACC(pred, label, num):
    count = 0
    for i in range(num):
        if pred.data[i][0] == label.data[i]:
            count += 1
    return count

class Trainer(object):
    def __init__(self, model,
                 train_iter, valid_iter, test_iter=None,
                 max_epochs=50):
        self.model = model
        self.optimizer = optim.Adamax(self.model.parameters(), lr = args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.max_epochs = max_epochs
        self.noutput = self.model.decoder.weight.size(0)

    def __forward(self, data, target):
        output = self.model(data)
        loss = self.criterion(output.view(-1, self.noutput), target)
        return output, loss

    def __train(self, lr, epoch):
        self.model.train()
        total_loss = 0
        acc = [0, 0]
        start_time = time.time()
        for batch, (d, targets) in enumerate(self.train_iter):
            self.model.zero_grad()
            output, loss = self.__forward(d, targets)
            count = countACC(torch.max(output, 1)[1], targets, args.batch_size)
            acc[0] += count
            acc[1] += args.batch_size
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | lr {:02.5f} | wps {:5.2f} | '
                        'loss {:5.2f} | acc {:1.3f}'.format(
                    epoch, lr,
                    args.batch_size / (elapsed / args.log_interval), cur_loss, float(acc[0])/acc[1]))
                total_loss = 0
                start_time = time.time()

    def train(self):
        # Loop over epochs.
        lr = args.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.max_epochs+1):
                epoch_start_time = time.time()
                self.__train(lr, epoch)
                val_loss = self.evaluate(self.valid_iter)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, (time.time() - epoch_start_time),))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    print("restore the model.")
                    model = torch.load(args.save)
                    lr *= args.lr_decay
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            self.model = torch.load(f)
        if not self.test_iter is None:
            self.evaluate(self.valid_iter)
            self.evaluate(self.test_iter, 'test')


    def evaluate(self, data_source, prefix='valid'):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        acc = [0, 0]
        for d, targets in data_source:
            output, loss = self.__forward(d, targets)
            count = countACC(torch.max(output, 1)[1], targets, 10)
            acc[0] += count
            acc[1] += 10
            total_loss += loss.data
        ave_loss = total_loss[0] / len(data_source)
        print('| {0} loss {1:5.2f} | {0} '.format(prefix, ave_loss))
        print('| acc {:1.3f}'.format(float(acc[0])/acc[1]))
        return ave_loss

if __name__ == '__main__':
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    path = args.data + '/'

    eval_batch_size = 10

    train_iter = DataIter(
        path + 'train.txt',
        args.batch_size,
        cuda = args.cuda,
    )
    valid_iter = DataIter(
        path + 'valid.txt',
        eval_batch_size,
        cuda = args.cuda,
    )
    test_iter = DataIter(
        path + 'test.txt',
        eval_batch_size,
        cuda = args.cuda,
    )

    ###############################################################################
    # Build the model
    ###############################################################################

    model = DNNModel(
        nfed = 20,
        nhid = args.nhid,
        noutputs = 201*201,
        dropout = args.dropout,
    )

    if args.cuda:
        model.cuda()


    trainer = Trainer(
        model = model,
        train_iter = train_iter,
        valid_iter = valid_iter,
        test_iter = test_iter,
        max_epochs = args.epochs
    )

    trainer.train()
