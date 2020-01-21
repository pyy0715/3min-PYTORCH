import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

vocab_size = 256
x_ = list(map(ord,'hello'))
y_ = list(map(ord, 'hola'))
x = torch.LongTensor(x_) #[5]
y = torch.LongTensor(y_) #[4]

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size) #[256, 16]
        self.encoder = nn.GRU(hidden_size, hidden_size) #[16, 16]
        self.decoder = nn.GRU(hidden_size, hidden_size) #[16, 16]
        self.project = nn.Linear(hidden_size, vocab_size) #[16, 256]

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
    
    def forward(self, inputs, targets):
        inital_state = self._init_state() #[numlayers: 1, batch: 1, hidden_size: 16]
        embedding = self.embedding(inputs).unsqueeze(1) # [5, 16] -> [seq_len: 5, batch: 1, embedding_size: 16]
        encoder_output, encoder_state = self.encoder(embedding, inital_state)
        # encoer_output = x, [5, 1, 16]
        # encoder_state = h, [1, 1, 16]
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([0])

        outputs = []
        for i in range(targets.size()[0]): #0~4
            decoder_input = self.embedding(decoder_input).unsqueeze(1) # [1, 1, 16]
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            projection = self.project(decoder_output) #[1, 256]
            outputs.append(projection)
            # Teacher Forching
            decoder_input = torch.LongTensor([targets[i]])

        outputs = torch.stack(outputs).squeeze()
        return outputs

seq2seq = Seq2Seq(vocab_size, 16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)

log = []
for i in range(1000):
    prediction = seq2seq(x, y)
    loss =  criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data 
    log.append(loss_val)
    if i % 100 ==0:
        print('\n Epoch: {}, Loss: {}'.format(i, loss_val.item()))
        _ , top1 = prediction.data.topk(1,1)
        print([chr(c) for c in top1.squeeze().numpy().tolist()])

plt.plot(log)
plt.ylabel('cross_entropy loss')
plt.show()