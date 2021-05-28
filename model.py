import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import math
from dataloader import dataloader, NUM_SEQUENCE

from fsa import ExecutionFSA, EOS, ACTION_SEP, NO_ARG

from alchemy_fsa import AlchemyFSA
from alchemy_world_state import AlchemyWorldState
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def execute(world_state, action_sequence):
    """Executes an action sequence on a world state.
    TODO: This code assumes the world state is a string. However, you may sometimes
    start with an AlchemyWorldState object. I suggest loading the AlchemyWorldState objects
    into memory in load_data, and moving that part of the code to load_data. The following
    code just serves as an example of how to 1) make an AlchemyWorldState and 2) execute
    a sequence of actions on it.
    Inputs:
        world_state (str): String representing an AlchemyWorldState.
        action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
            (like in the JSON file).
    """
    alchemy_world_state = AlchemyWorldState(world_state)
    fsa = AlchemyFSA(alchemy_world_state)

    for action in action_sequence:
        split = action.split(" ")
        act = split[0]
        arg1 = split[1]

        # JSON file doesn't contain  NO_ARG.
        if len(split) < 3:
            arg2 = NO_ARG
        else:
            arg2 = split[2]

        fsa.feed_complete_action(act, arg1, arg2)

    return fsa.world_state()

class world_state_encoder(nn.Module):
    def __init__(self, pos_embedded_dim=10, color_embedded_dim=10, num_layers=1, hidden_dim=10):
        super(world_state_encoder, self).__init__()
        self.pos_embedded_dim = pos_embedded_dim
        self.color_embedded_dim = color_embedded_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.color_embedding = nn.Embedding(7, self.color_embedded_dim)
        self.pos_embedding = nn.Embedding(7, self.pos_embedded_dim)
        self.lstm = nn.LSTM(
                self.color_embedded_dim, self.hidden_dim, num_layers=self.num_layers, batch_first = True
            )
    
    def forward(self, X):
        batch_size = X.shape[0]
        beaker_id = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
        beaker_id = beaker_id.repeat((batch_size, 1)).to(device)
        beaker_id = self.pos_embedding(beaker_id)
        world_state = None
        all_colors = None
        for i in range(batch_size):
            colors = None
            for j in range(0, X.shape[1], 5):
                processed_color = self.color_embedding(X[i][j+1:j+5]).reshape((1, 4, -1))
                _, (encoded_color, _) = self.lstm(processed_color)
                colors = encoded_color if j == 0 else torch.cat((colors, encoded_color), dim=1)
            all_colors = colors if i == 0 else torch.cat((all_colors, colors), dim=0)
        all_colors = all_colors.to(device)
        context = torch.cat((beaker_id, all_colors), dim=2)
        context = torch.reshape(context, (batch_size, -1))
        return context

class instruction_encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=100, embedded_size=50, num_layers=1, bidirectional=True):
        super(instruction_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=0)
        self.lstm = nn.LSTM(embedded_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, X, valid_length):
        from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
        X = self.embedding(X)
        X = pack_padded_sequence(X, valid_length, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(X)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        return output

class attention_action_decoder(nn.Module):
    def __init__(self, action_size, input_size, ins_hidden_size, hidden_size, embedding_size, env_dim, num_layers=1):
        super(attention_action_decoder, self).__init__()
        self.embedding = nn.Embedding(action_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_to_action = nn.Linear(hidden_size, action_size)
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.hidden_to_action.weight)
        # init W for h_i, W, h^q
        self.W_c = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2 * ins_hidden_size, hidden_size)))
        self.W_p = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2 * ins_hidden_size, 2 * ins_hidden_size + hidden_size)))
        self.W_s_b_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        self.W_s_b_2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        self.W_s_c_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        self.W_s_c_2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))

        # weight matrix for lstm input
        W_d_dim = 4 * env_dim + 4 * ins_hidden_size + embedding_size

        self.W_b_d = nn.Linear(W_d_dim, input_size)
        nn.init.xavier_uniform_(self.W_b_d.weight)

    def attend(self, H, query, weight):
        """
        H: batch_size x sen_length x 2*hidden
        weight: 2 * hidden x hidden
        query:  batch_size x hidden
        """
        query = query.to(device)
        weight = weight.to(device)
        alpha = torch.matmul(H, weight)
        extend_query = (query.reshape((query.shape[0], 1, query.shape[1]))).repeat(1, alpha.shape[1], 1)
        alpha = torch.einsum('ijk, ijk -> ijk', [alpha, extend_query])
        alpha = alpha.sum(-1)
        alpha = torch.nn.Softmax(1)(alpha)
        alpha = torch.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))
        alpha = alpha.repeat((1, 1, H.shape[2]))
        z = torch.einsum('ijk, ijk -> ijk', [alpha, H])
        z = z.sum(1)
        return z

    def init_hidden(self, batch_size):
        h_0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros((batch_size, self.hidden_size)))).to(device)
        c_0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros((batch_size, self.hidden_size)))).to(device)
        return (h_0, c_0)

    def forward(self, ins, his, actions, current_env_context, ini_env_context, ins_valid, teacher_force=True):
        # from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
        batch_size = ins.shape[0]
        h, c = self.init_hidden(batch_size)
        all_outputs = None
        X = self.embedding(actions.to(device))
        current_env_context = torch.reshape(current_env_context, (current_env_context.shape[0], 1, current_env_context.shape[1]))
        current_env_context = current_env_context.repeat((1, X.shape[1], 1))
        ini_env_context = torch.reshape(ini_env_context, (ini_env_context.shape[0], 1, ini_env_context.shape[1]))
        ini_env_context = ini_env_context.repeat((1, X.shape[1], 1))
        tanh = nn.Tanh()
        act_len = X.shape[1]

        for i in range(act_len):
            z_k_c = self.attend(ins, h, self.W_c)
            z_k_p = self.attend(his, torch.cat((h, z_k_c), 1), self.W_p)
            z_s_1_k_1 = self.attend(
                ini_env_context, torch.cat((h, z_k_c), 1), self.W_s_b_1)
            z_s_1_k_2 = self.attend(
                ini_env_context, torch.cat((h, z_k_c), 1), self.W_s_b_2)
            z_s_1_k = torch.cat((z_s_1_k_1, z_s_1_k_2), -1)
            z_s_k_k_1 = self.attend(
                current_env_context, torch.cat((h, z_k_c), 1), self.W_s_c_1)
            z_s_k_k_2 = self.attend(
                current_env_context, torch.cat((h, z_k_c), 1), self.W_s_c_2)
            z_s_k_k = torch.cat((z_s_k_k_1, z_s_k_k_2), -1)
            phi_action_i = X.permute(1, 0, 2)[i]
            h_k_1 = torch.cat(
                (z_k_c, z_k_p, z_s_1_k, z_s_k_k, phi_action_i), -1)
            h_k = tanh(self.W_b_d(h_k_1))
            h_k = torch.reshape(h_k, (h_k.shape[0], 1, h_k.shape[1]))
            h = torch.reshape(h, (1, h.shape[0], h.shape[1]))
            c = torch.reshape(c, (1, c.shape[0], c.shape[1]))
            output, (h, c) = self.lstm(h_k, (h, c))
            h = h[0]
            c = c[0]
            if i == 0:
                all_outputs = output
            else:
                all_outputs = torch.cat((all_outputs, output), dim=1)
        all_outputs = self.hidden_to_action(all_outputs)
        if teacher_force:
            return all_outputs

        softmax = nn.Softmax(dim=0)
        return torch.argmax(softmax(all_outputs[0][0]))

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, action_size, ins_hidden_size, ins_embedding_size, act_embedding_size, act_input_size, act_hidden_size, pos_embedding_size, color_embedding_size, color_hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = instruction_encoder(vocab_size, hidden_size=ins_hidden_size, embedded_size=ins_embedding_size, num_layers=1, bidirectional=True)
        self.env_encoder = world_state_encoder(pos_embedded_dim=pos_embedding_size, color_embedded_dim=color_embedding_size, hidden_dim=color_hidden_size)
        env_dim = 7 * (pos_embedding_size + color_hidden_size)
        self.decoder = attention_action_decoder(action_size, act_input_size, ins_hidden_size, act_hidden_size, act_embedding_size, env_dim)
    
    def train(self, dl, batch_size=32, epoch=10, learning_rate=0.01):
        loss_function = MaskedSoftmaxCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in range(epoch):
            avg_loss = 0
            batch_count = 0
            for step, batch in enumerate(dl):
                batch_count += 1
                batch = tuple(t.to(device) for t in batch)
                ins, his_ins, ins_valid, his_valid, ini_env, current_env, act_id, valid_act, y_true, y_true_valid = batch
                optimizer.zero_grad()
                ins_out = self.encoder(ins, ins_valid)
                his_out = self.encoder(his_ins, his_valid)
                ini_env_context = self.env_encoder(ini_env)
                current_env_context = self.env_encoder(current_env)
                pred = self.decoder(ins_out, his_out, act_id, current_env_context, ini_env_context, ins_valid, teacher_force=True)
                l = loss_function(pred, y_true.to(device), y_true_valid)
                l.sum().backward()
                if step % 20 == 0:
                    print("batch_loss:", l.sum().item() / batch_size)
                avg_loss = avg_loss + l.sum().item() / batch_size
                optimizer.step()
            print("loss at epoch " + str(i) + ":")
            print(avg_loss / batch_count)
    
    # def predict(self, dl):


# def save_output(result):


def main():
    train = "train.json"
    dev = "dev.json"
    test = "test_leaderboard.json"
    batch_size = 32
    num_filter = 10
    DL = dataloader(train, dev, test, batch_size, num_filter)
    train_loader = DL.train_loader()

    vocab_size = len(DL.instruction_vocab)
    action_size = len(DL.action_vocab)

    # instruction encoder setting
    ins_hidden_size = 100
    ins_embedding_size = 50
    ins_num_lstm_layers = 1

    # world state encoder setting
    pos_embedding_size = 10
    color_embedding_size = 10
    color_hidden_size = 20

    # action setting
    act_embedding_size = 50
    act_input_size = 100
    act_hidden_size = 100

    model = Seq2Seq(vocab_size, action_size, ins_hidden_size, ins_embedding_size, act_embedding_size, act_input_size, act_hidden_size, pos_embedding_size, color_embedding_size, color_hidden_size)
    model.to(device)
    epoch = 10
    learning_rate = 0.001
    model.train(train_loader, batch_size, epoch, learning_rate)
    # dev_pred = model.predict()


if __name__ == "__main__":
    main()