class LSTM_MicMac(nn.Module):
    def __init__(self, input_size_rnn, hidden_size, output_size_rnn, dropout = 0.2):
        '''
        DNN model with long short term memory (LSTM) gates to process sequential data
        @args input_size_rnn: size of one element in a sequence of inputs.
        @args hidden_size: size of hidden and cell kernels
        @args output_size: size of expected output
        '''
        super(LSTM_MicMac, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.dropout = nn.Dropout(p=dropout)

        # 2-recurrent-layers LSTM network
        self.lstm_mic = nn.LSTM(input_size = input_size_rnn, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, dropout=dropout)
        self.lstm_mac = nn.LSTM(input_size = input_size_rnn, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, dropout=dropout)
        
        # dense linear layer on macro and micro lstm outputs
        last_layer_size = int(hidden_size/2)
        self.linear_mic = nn.Linear(hidden_size, last_layer_size, bias = True)
        self.linear_mac = nn.Linear(hidden_size, last_layer_size, bias = True)
        # dense linear layer to output classification
        self.linear = nn.Linear(last_layer_size + last_layer_size, output_size_rnn, bias = True)
        self.relu = nn.ReLU()
        
    def isCuda(self):
        return next(self.linear.parameters()).is_cuda
        
    def initHiddenAndCell(self, batch_size):
        '''
        initialize hidden and cell gates with zeros
        @args batch_size: the number of input sequences in the batch
        '''
        h_t = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c_t = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if self.isCuda():
            h_t = h_t.cuda()
            c_t = c_t.cuda()
        return (h_t, c_t)
                       
    def forward(self, X_mic, X_mac):
        '''
        apply LSTM and linear kernels
        @args X: array (sequence_length x batch_size x input_size)
        '''
        if self.isCuda():
            X_mic = X_mic.cuda()
            X_mac = X_mac.cuda()
        self.hidden_state_mic = self.initHiddenAndCell(X_mic.size(1))
        self.hidden_state_mac = self.initHiddenAndCell(X_mac.size(1))
        self.inputs_mic = self.dropout(X_mic)
        self.inputs_mac = self.dropout(X_mac)
        self.lstm_outs_mic, self.hidden_state_mic = self.lstm_mic(self.inputs_mic, self.hidden_state_mic)
        self.lstm_outs_mac, self.hidden_state_mac = self.lstm_mac(self.inputs_mac, self.hidden_state_mac)
        # only get last input
        self.lstm_outs_mic = self.lstm_outs_mic[-1,:,:]
        self.lstm_outs_mac = self.lstm_outs_mac[-1,:,:]
        # note: view format ([e_seq_len v1, ..., e_seq_len v_batch_size])
        self.outs_mic = self.relu(self.linear_mic(self.lstm_outs_mic.view(-1, self.hidden_size)))
        self.outs_mac = self.relu(self.linear_mac(self.lstm_outs_mac.view(-1, self.hidden_size)))
        # concatenate both lstm outputs
        self.merge = torch.cat([self.outs_mic, self.outs_mac], dim = 1)
        self.merge = self.dropout(self.merge)
        self.out = self.linear(self.merge)
        return self.out