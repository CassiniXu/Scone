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
