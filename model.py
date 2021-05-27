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
    def __init__(
        self,
        pos_embedded_dim=10,
        color_embedded_dim=10,
        num_layers=1,
        hidden_dim=10,
    ):
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
                _, (encoded_color, _) = self.lstm(self.color_embedding(X[i][j+1:j+5]).reshape((1, -1)))
                colors = all_colors if j == 0 else torch.cat((colors, encoded_color), dim=1)
            all_colors = colors if i == 0 else torch.cat((all_colors, colors), dim=0)
        all_colors = all_colors.to(device)
        context = torch.cat((beaker_id, all_colors), dim=2)
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
        self.hidden_to_action = nn.Linear(hidden_size, acton_size)
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
    
    def forward(self, ins, his, actions, current_env_context, ini_env_context, ins_valid, teacher_force=True):
        from torch


        

def main():
    train = "train.json"
    dev = "dev.json"
    test = "test_leaderboard.json"
    DL = dataloader(train, dev, test)
    train_loader = DL.train_loader() # 

    # ins, his_ins, ins_valid, his_invalid, ini_env, current_env, act_id, valid_act)
    


if __name__ == "__main__":
    main()