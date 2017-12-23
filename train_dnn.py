import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

parser = argparse.ArgumentParser(description='PyTorch Stock Value Prediction Model')
parser.add_argument('--data', type=str, default='./data/sz002821_2',
                    help='location of the data')
parser.add_argument('--nfeatures', type=int, default=30,
                    help='dimension of features')
parser.add_argument('--contexts', type=int, default=5,
                    help='size of the context')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--lr_decay', type=float, default=0.1,
                    help='decay lr by the rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
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
    def __init__(self, path, batch_size, scaler, cuda=False):
        self.path = path
        self.batch_size = batch_size
        self.cuda = cuda
        self.scaler = scaler

        self.build_data()
        self.batchify()

    def build_data(self):
        #data_type = np.dtype([('features1', 'f8', (19, )), ('features2', 'f8', (11, )),('labels1', 'i8', (1, )), ('labels2', 'i8', (1, ))])
        data_type = np.dtype([('features1', 'f8', (30, )), ('labels1', 'i8', (1, )), ('labels2', 'i8', (1, ))])
        
        data = np.loadtxt(self.path, data_type, delimiter=' ')
        features1 = data['features1']
        #features2 = data['features2']
        labels1 = data['labels1']
        labels2 = data['labels2']
        #features1 = 0.2 * (features1 - 130)
        #np.concatenate((features1, features2, features1 * features2), axis=1)
        if self.scaler == None:
            self.scaler = StandardScaler().fit(features1)
        features1 = self.scaler.transform(features1)
        features = features1
        #features = np.concatenate((features1, features2), axis=1)
        count0 = 0
        count1 = 0
        count2 = 0
        for i in range(labels1.shape[0]):
            if labels1[i] + labels2[i] > 0:
                labels1[i] = 2
                count2 += 1
            elif labels1[i] + labels2[i] == 0:
                labels1[i] = 1
                count1 += 1
            else:
                labels1[i] = 0
                count0 += 1
        count = float(count0+count1+count2)
        print "Class 0: %.4f%%, Class 1: %.4f%%, Class 2: %.4f%%"%(count0/count*100, count1/count*100, count2/count*100)
            #labels1[i]  = labels1[i] *201 + labels2[i]
        features = torch.Tensor(features)
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
        data = data.view(-1, self.batch_size, args.nfeatures).contiguous()
        label = label.view(-1, self.batch_size).contiguous()
        self.data = data.cuda() if self.cuda else data
        self.label = label.cuda() if self.cuda else label

    def __iter__(self):
        for idx in range(0, self.data.size(0) - 1, 1):
            data = []
            for cur_idx in range(idx+1-args.contexts, idx+1):
                if cur_idx < 0:
                    data.append(np.zeros((self.batch_size, args.nfeatures)))
                else:
                    data.append(self.data[cur_idx].cpu().numpy())
            data = np.concatenate(data, axis=1)
            data = Variable(torch.Tensor(data))
            data = data.cuda() if self.cuda else data
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

def count(pred, label, num, results, labels):
    count = 0
    pred = pred.cpu()
    label = label.cpu()
    for i in range(num):
        results.append(pred.data[i][0])
        labels.append(label.data[i])

class Trainer(object):
    def __init__(self, model,
                 train_iter, valid_iter, test_iter=None,
                 max_epochs=50):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.max_epochs = max_epochs
        self.noutput = self.model.decoder.weight.size(0)

    def score(self):
        acc = accuracy_score(self.labels, self.results)
        print "total acc: %.4f%%"%(acc * 100)
        for i in range(3):
            pre = precision_score(self.labels, self.results, labels=[i], average='micro')
            rec = recall_score(self.labels, self.results, labels=[i], average='micro')
            f1 = f1_score(self.labels, self.results, labels=[i], average='micro')
            print "for class %d:"%(i)
            print "precision: %.4f, recall: %.4f, f1: %.4f "%(pre, rec, f1)
            print ""
        return acc

    def clear_results(self):
        self.results = []
        self.labels = []
        return

    def __forward(self, data, target):
        output = self.model(data)
        loss = self.criterion(output.view(-1, self.noutput), target)
        return output, loss

    def __train(self, lr, epoch):
        self.model.train()
        #self.clear_results()
        total_loss = 0
        start_time = time.time()
        for batch, (d, targets) in enumerate(self.train_iter):
            self.model.zero_grad()
            output, loss = self.__forward(d, targets)
            #count(torch.max(output, 1)[1], targets, args.batch_size, self.results, self.labels)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | lr {:02.5f} | wps {:5.2f} | '
                        'loss {:5.2f} '.format(
                    epoch, lr,
                    args.batch_size / (elapsed / args.log_interval), cur_loss))
                #self.score()
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
                if not best_val_loss or val_loss > best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    print("restore the model.")
                    model = torch.load(args.save)
                    lr *= args.lr_decay
                    self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
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
        self.clear_results()
        total_loss = 0
        for d, targets in data_source:
            output, loss = self.__forward(d, targets)
            count(torch.max(output, 1)[1], targets, 10, self.results, self.labels)
            total_loss += loss.data
        ave_loss = total_loss[0] / len(data_source)
        print('| {0} loss {1:5.2f} | {0} '.format(prefix, ave_loss))
        acc = self.score()
        return acc

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

    scaler = None
    train_iter = DataIter(
        path + 'train.txt',
        args.batch_size,
        scaler,
        cuda = args.cuda,
    )
    scaler = train_iter.scaler

    valid_iter = DataIter(
        path + 'valid.txt',
        eval_batch_size,
        scaler,
        cuda = args.cuda,
    )
    test_iter = DataIter(
        path + 'test.txt',
        eval_batch_size,
        scaler,
        cuda = args.cuda,
    )

    ###############################################################################
    # Build the model
    ###############################################################################

    model = DNNModel(
        nfed = args.nfeatures * args.contexts,
        nhid = args.nhid,
        noutputs = 3,
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
