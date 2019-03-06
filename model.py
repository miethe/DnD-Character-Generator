import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNNCellModel(nn.Module):
    """
    Character-based RNN model using LSTMCell as it's core. Since LSTMCell takes input at specific time-step,
    looping over time-steps is done in training method.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx, cx):
        """
        :param input: Input tensor with shape (batch_size, input_size)
        :param hx: Previous hidden state tensor with shape (batch_size, hidden_size)
        :param cx: Previous memory state tensor with shape (batch_size, hidden_size)
        :return:
            logits: Output from dense layer tensor with shape (batch_size, output_size)
            hx: Output hidden state tensor with shape (batch_size, hidden_size)
            cx: Output memory state tensor with shape (batch_size, hidden_size)
        """
        hx, cx = self.lstm_cell(input, (hx, cx))
        logits = self.dense(self.dropout(hx))

        return logits, hx, cx

    def init_states(self, batch_size, device):
        hx = torch.zeros(batch_size, self.hidden_size).to(device)
        cx = torch.zeros(batch_size, self.hidden_size).to(device)

        return hx, cx


class RNNLayerModel(nn.Module):
    """
    Character-based RNN using LSTMLayer as it's core. Shows how packing and unpacking of sequences can be done in
    PyTorch if you (un)avoidably have variable length sequences.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hx, cx, lengths):
        """
        :param inputs: Input tensor with shape (max_length, batch_size, input_size)
        :param hx: Previous hidden state tensor with shape (num_layers, batch_size, hidden_size)
        :param cx: Previous memory state tensor with shape (num_layers, batch_size, hidden_size)
        :param lengths: Tensor containing length for each sample in batch with shape (batch_size)
        :return:
            logits: Output from dense layer tensor with shape (max_length, batch_size, output_size)
            hx: Output hidden state tensor with shape (num_layers, batch_size, hidden_size)
            cx: Output memory state tensor with shape (num_layers, batch_size, hidden_size)
        """
        inputs = pack_padded_sequence(inputs, lengths=lengths)
        outputs, (h_n, c_n) = self.lstm(inputs, (hx, cx))
        pad_outputs, _ = pad_packed_sequence(outputs)
        logits = self.dense(self.dropout(pad_outputs))

        return logits, h_n, c_n

    def init_states(self, batch_size, device):
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return hx, cx
