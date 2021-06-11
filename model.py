import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import math
from dataloader import dataloader, NUM_SEQUENCE, UNKNOWN, PAD, START, NUM_CHEMICAL_LAYERS, color_to_id, id_to_color
from fsa import ExecutionFSA, EOS, ACTION_SEP, NO_ARG

from alchemy_fsa import AlchemyFSA
from alchemy_world_state import AlchemyWorldState
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import os
import pandas as pd

NUM_COLORS = 7
NUM_POS = 7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class world_state_encoder(nn.Module):
    def __init__(self, pos_embedded_dim=20, color_embedded_dim=20, num_layers=1, hidden_dim=10):
        super(world_state_encoder, self).__init__()
        self.color_embedding = nn.Embedding(NUM_COLORS, color_embedded_dim)
        self.pos_embedding = nn.Embedding(NUM_POS, pos_embedded_dim)
        self.lstm = nn.LSTM(color_embedded_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    
    def forward(self, X):
        batch_size = X.shape[0]
        beaker_id = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device).repeat((batch_size, 1))
        beaker_id = self.pos_embedding(beaker_id)
        all_colors = self.color_embedding(torch.reshape(X, (-1, NUM_CHEMICAL_LAYERS)))
        _, (encoded_color, _) = self.lstm(all_colors)
        encoded_color = torch.reshape(encoded_color[0], (batch_size, NUM_POS, -1)).to(device)
        context = torch.cat((beaker_id, encoded_color), dim=2)
        context = context.reshape(batch_size, -1)
        return context

class instruction_encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=100, embedded_size=50, num_layers=1, bidirectional=True, p=0.05):
        super(instruction_encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=0)
        self.lstm = nn.LSTM(embedded_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, X, valid_length=None, is_predict=False):
        if is_predict: valid_length = torch.tensor([int(X.shape[1])])
        X = self.embedding(X)
        # X = self.dropout(X)
        X = pack_padded_sequence(X, valid_length.to(torch.device("cpu")), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(X)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        hidden = torch.cat((hidden[0], hidden[1]), dim=-1)
        return hidden

class attention_action_decoder(nn.Module):
    def __init__(self, action_size, input_size, ins_hidden_size, hidden_size, embedding_size, env_dim, num_layers=1, p=0.05):
        super(attention_action_decoder, self).__init__()
        input_size = 2 * env_dim + 4 * ins_hidden_size + embedding_size
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(action_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.hidden_to_action = nn.Linear(hidden_size, action_size)
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.hidden_to_action.weight)
        # init W for h_i, W, h^q
        # self.W_c = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2 * ins_hidden_size, hidden_size)))
        # self.W_p = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2 * ins_hidden_size, 2 * ins_hidden_size + hidden_size)))
        # self.W_s_b_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        # self.W_s_b_2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        # self.W_s_c_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))
        # self.W_s_c_2 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size + 2 * ins_hidden_size)))

        # weight matrix for lstm output
        # W_d_dim = 4 * env_dim + 4 * ins_hidden_size + embedding_size
        W_d_dim = env_dim + 2 * ins_hidden_size + embedding_size

        self.W_b_d = nn.Linear(W_d_dim, input_size)
        nn.init.xavier_uniform_(self.W_b_d.weight)

        self.W_env = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(env_dim, hidden_size)))

    def attend(self, H, query, weight):
        """
        H: batch_size x sen_length x 2*ins_hidden
        weight: 2 * ins_hidden x hidden
        query:  batch_size x hidden
        """
        H_w = torch.matmul(H, weight)
        alpha_bar = torch.einsum('ijk, ik -> ij', [H_w, query])
        alpha = nn.Softmax(1)(alpha_bar)
        c = torch.einsum('ijk, ij -> ik', [H, alpha])
        return c

    def init_hidden(self, batch_size):
        h_zeros = torch.zeros(batch_size, self.hidden_size, device = device)
        c_zeros = torch.zeros(batch_size, self.hidden_size, device = device)
        return h_zeros, c_zeros

    def forward(self, ins, his, actions, current_env_context, ini_env_context, h=None, c=None, teacher_force=True):
        X = self.embedding(actions.to(device))
        # X = self.dropout(X)
        batch_size = ins.shape[0]
        if teacher_force or (h == None and c == None):
            h, c = self.init_hidden(batch_size)
        act_len = X.shape[1]
        output = []
        X = X.permute(1, 0, 2)
        tanh = nn.Tanh()
        for i in range(act_len):
            """
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
            """
            # enc_ins = self.attend(ins, h, self.W_c)
            # enc_his = self.attend(ins, h, self.W_c)
            # enc_env = self.attend(current_env_context, h, self.W_env)
            # enc_env = current_env_context
            phi_action_i = X[i]
            # h_k = tanh(self.W_b_d(torch.cat((ins, current_env_context, phi_action_i), -1)))
            # h_k = self.W_b_d(torch.cat((ins, current_env_context, phi_action_i), -1))
            h_k = torch.cat((ins, current_env_context, his, ini_env_context, phi_action_i), -1)
            h, c = self.lstm(h_k, (h, c))
            output.append(h)
        output = torch.stack(output, dim=0)
        output = self.hidden_to_action(output)
        output = output.permute(1, 0, 2)
        if teacher_force:
            return output
        softmax = nn.Softmax(dim=0)
        return torch.argmax(softmax(output[0][0])), h, c

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, action_size, ins_hidden_size, ins_embedding_size, act_embedding_size, act_input_size, act_hidden_size, pos_embedding_size, color_embedding_size, color_hidden_size, p):
        super(Seq2Seq, self).__init__()
        self.encoder = instruction_encoder(vocab_size, hidden_size=ins_hidden_size, embedded_size=ins_embedding_size, num_layers=1, bidirectional=True, p=p)
        self.env_encoder = world_state_encoder(pos_embedded_dim=pos_embedding_size, color_embedded_dim=color_embedding_size, hidden_dim=color_hidden_size)
        env_dim = 7 * (pos_embedding_size + color_hidden_size)
        self.decoder = attention_action_decoder(action_size, act_input_size, ins_hidden_size, act_hidden_size, act_embedding_size, env_dim, p=p)
    
    def train(self, dl, batch_size=32, epoch=10, learning_rate=0.01):
        loss_function = MaskedSoftmaxCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
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
                pred = self.decoder(ins_out, his_out, act_id, current_env_context, ini_env_context, teacher_force=True)
                l = loss_function(pred, y_true.to(device), y_true_valid)
                l.sum().backward()
                if step % 20 == 0:
                    print("batch_loss:", l.sum().item() / batch_size)
                avg_loss = avg_loss + l.sum().item() / batch_size
                optimizer.step()
            print("loss at epoch " + str(i) + ":")
            print(avg_loss / batch_count)
    
    def clean_action(self, action, action_split=False):
        valid_action = []
        i = 0
        while i < len(action):
            if action_split == False:
                if action[i] == START or action[i] == PAD or action[i] == EOS:
                    i += 1
                else:
                    split_action = action[i].split(" ")
                    if split_action[0] == "pop":
                        split_action.append(NO_ARG)
                    valid_action.append(split_action)
                    i += 1
            else:
                """
                not implemented
                """
        return valid_action

    def delete_extra_chemicals(self, ws):
        ws = ws.split(" ")
        cleaned_ws = ""
        for i in ws:
            cleaned_ws += i[:6]
            cleaned_ws += " "
        return cleaned_ws[:-1]

    def recover_ws(self, ws):
        str_ws = ""
        for i in range(0, len(ws), NUM_CHEMICAL_LAYERS):
            str_ws += " "
            str_ws = str_ws + str(int(i / NUM_CHEMICAL_LAYERS) + 1) + ":" + id_to_color[ws[i]]

            if id_to_color[ws[i+1]] != "_":
                str_ws += id_to_color[ws[i+1]]
            else:
                continue
            if id_to_color[ws[i+2]] != "_":
                str_ws += id_to_color[ws[i+2]]
            else:
                continue
            if id_to_color[ws[i+3]] != "_":
                str_ws += id_to_color[ws[i+3]]
            else:
                continue
        return str_ws[1:]

    def predict(self, ins, his, ini_env, act_ix, ix_act, DL, max_act_len = 8):
        data_length = len(ins)

        curr_env = ini_env[0]
        pred_act = START
        ws = self.recover_ws(ini_env[0][0].tolist())

        all_ws = []

        for index in range(data_length):
            act_sequence = []


            encoded_ins = self.encoder(ins[index], is_predict=True)
            encoded_his = self.encoder(his[index], is_predict=True)

            encoded_ini = self.env_encoder(ini_env[index])

            encoded_curr = encoded_ini
            if index % NUM_SEQUENCE != 0:
                encoded_curr = self.env_encoder(curr_env)

            
            if index % NUM_SEQUENCE == 0:
                ws = self.recover_ws(ini_env[index][0].tolist())

            pred_act = START
            pred_count = 0

            h = None
            c = None
            while pred_act != EOS:
                pred_act = torch.tensor([[act_ix[pred_act]]], dtype=torch.long)

                pred_act, h, c = self.decoder(encoded_ins, encoded_his, pred_act, encoded_curr, encoded_ini, h, c, teacher_force=False)

                pred_act = ix_act[pred_act.item()]
                act_sequence.append(pred_act)
                pred_count += 1
                if pred_count >= max_act_len:
                    break
            
            act_sequence = self.clean_action(act_sequence)
            ws = AlchemyWorldState(ws).execute_seq(act_sequence).__str__()
            ws = self.delete_extra_chemicals(ws)
            all_ws.append(ws)
            curr_env = DL.pad_ws(ws)

        print("Predictions complete.")
        return all_ws


def save_output(ws, idf, ins_file, inter_file):
    ins_df = pd.DataFrame()
    inter_df = pd.DataFrame()
    ins = []
    ins_id = []
    inter = []
    inter_id = []

    for i in range(len(ws)):
        ins.append(ws[i])
        ins_id.append(idf[i])
        if i % NUM_SEQUENCE == (NUM_SEQUENCE-1):
            inter.append(ws[i])
            inter_id.append(idf[i][:-2])
    ins_df["id"] = ins_id
    ins_df["final_world_state"] = ins
    inter_df["id"] = inter_id
    inter_df["final_world_state"] = inter
    ins_df.to_csv(ins_file, index=False)
    inter_df.to_csv(inter_file, index=False)
    print("Results saved.")

def main():
    train = "train.json"
    dev = "dev.json"
    test = "test_leaderboard.json"
    batch_size = 32
    num_filter = 2
    DL = dataloader(train, dev, test, batch_size, num_filter)
    train_loader = DL.train_loader()

    vocab_size = len(DL.instruction_vocab)
    action_size = len(DL.action_vocab)

    # instruction encoder setting
    ins_hidden_size = 100
    ins_embedding_size = 50
    ins_num_lstm_layers = 1

    # world state encoder setting
    pos_embedding_size = 25
    color_embedding_size = 25
    color_hidden_size = 75


    # action setting
    act_embedding_size = 50
    act_input_size = 1500
    act_hidden_size = 200
    p = 0.05

    model = Seq2Seq(vocab_size, action_size, ins_hidden_size, ins_embedding_size, act_embedding_size, act_input_size, act_hidden_size, pos_embedding_size, color_embedding_size, color_hidden_size, p)
    model.to(device)
    # from prettytable import PrettyTable
    # table = PrettyTable(["Modules", "Parameters"])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
    #     param = parameter.numel()
    #     table.add_row([name, param])
    #     total_params+=param
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    # return total_params
    epoch = 17
    learning_rate = 1e-3
    model.train(train_loader, batch_size, epoch, learning_rate)
    dev_ins, dev_his, dev_ini_env, dev_id = DL.dev_data()
    result = model.predict(dev_ins, dev_his, dev_ini_env, DL.actions_to_id, DL.id_to_actions, DL, max_act_len=10)
    result_ins_file = "dev_instruction_pred.csv"
    result_inter_file = "dev_inter_pred.csv"
    save_output(result, dev_id, result_ins_file, result_inter_file)


if __name__ == "__main__":
    main()