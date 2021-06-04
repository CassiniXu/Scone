import numpy as np
import json
from fsa import EOS, ACTION_SEP, NO_ARG
import torch
import string
UNKNOWN = "<UNKNOWN>"
PAD = "<PAD>"
START = "<START>"
NUM_SEQUENCE = 5
NUM_CHEMICAL_LAYERS = 4
color_to_id = {"_": 0, "y": 1, "o": 2, "g": 3, "r": 4, "b": 5, "p": 6}
id_to_color = {0: "_", 1: "y", 2: "o", 3: "g", 4: "r", 5: "b", 6: "p"}
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class dataloader():
    def __init__(self, train, dev, test, batch_size = 32, num_filter = 10):
        self.instructions_to_id = None
        self.actions_to_id = None
        self.id_to_actions = None
        self.train = train
        self.dev = dev
        self.test = test
        self.batch_size = batch_size
        instructions, his_instructions, actions, initial_environments, environments, identifiers = self.load_data(self.train)
        self.construct_vocab(actions, instructions, num_filter)
        dev_ins, dev_his, _, dev_ini_env, _, dev_id = self.load_data(self.dev)
        test_ins, test_his, _, test_ini_env, _, test_id = self.load_data(self.test)

        """
        prepare for training data
        """

        # replace with id
        train_id = self.replace_with_id(instructions)
        train_his_id = self.replace_with_id(his_instructions)
        act_id = self.replace_with_id(actions, False)

        """
        construct ground truth act
        """
        act_id, ground_act_id = self.construct_act(act_id)



        # padding
        train_id_pad, train_valid_length = self.padding(train_id)
        train_his_id_pad, train_his_valid_length = self.padding(train_his_id)
        act_id_pad, act_valid_length = self.padding(act_id)
        ground_act_id_pad, ground_act_id_pad_valid_length = self.padding(ground_act_id)

        # to tensor
        train_id_pad = torch.tensor(train_id_pad,  dtype=torch.long)
        train_his_id_pad = torch.tensor(train_his_id_pad,  dtype=torch.long)
        train_valid_length = torch.tensor(train_valid_length, dtype=torch.long)
        train_his_valid_length = torch.tensor(train_his_valid_length, dtype=torch.long)


        act_id_pad = torch.tensor(act_id_pad, dtype=torch.long)
        act_valid_length = torch.tensor(act_valid_length, dtype=torch.long)

        ground_act_id_pad = torch.tensor(ground_act_id_pad, dtype=torch.long)
        ground_act_id_pad_valid_length = torch.tensor(ground_act_id_pad_valid_length, dtype=torch.long)

        # process and replace world state
        initial_environments = self.process_world_state(initial_environments)
        environments = self.process_world_state(environments)
        initial_environments = self.replace_world_state(initial_environments)
        environments = self.replace_world_state(environments)
        initial_environments = torch.tensor(initial_environments, dtype=torch.long)
        environments = torch.tensor(environments, dtype=torch.long)

        self.train_dataloader = self.construct_dataloader(train_id_pad, train_his_id_pad, train_valid_length, train_his_valid_length, initial_environments, environments, act_id_pad, act_valid_length, ground_act_id_pad, ground_act_id_pad_valid_length)

        """
        process dev and test data
        """
        dev_ins, dev_his, dev_ini_env = self.process_non_train_ins(dev_ins, dev_his, dev_ini_env)
        test_ins, test_his, test_ini_env = self.process_non_train_ins(test_ins, test_his, test_ini_env)

        # return
        self.dev_ins = dev_ins
        self.dev_his = dev_his
        self.dev_ini_env = dev_ini_env
        self.dev_id = dev_id
        self.test_ins = test_ins
        self.test_his = test_his
        self.test_ini_env = test_ini_env
        self.test_id = test_id

    def construct_act(self, act):
        input_act = []
        ground_act = []
        for i in range(len(act)):
            t_input_act = []
            t_ground_act = []
            for j in range(len(act[i]) - 1):
                t_input_act.append(act[i][j])
                t_ground_act.append(act[i][j+1])
            input_act.append(t_input_act)
            ground_act.append(t_ground_act)
        return input_act, ground_act

    def train_loader(self):
        return self.train_dataloader
    
    def dev_data(self):
        return self.dev_ins, self.dev_his, self.dev_ini_env, self.dev_id
    
    def test_data(self):
        return self.test_ins, self.test_his, self.test_ini_env, sef.test_id

    def process_non_train_ins(self, ins, his, ini_env):
        ins = self.replace_with_id(ins)
        his = self.replace_with_id(his)
        ini_env = self.process_world_state(ini_env)
        ini_env = self.replace_world_state(ini_env)
        data_length = len(ins)
        for i in range(data_length):
            ins[i] = torch.tensor([ins[i]], dtype=torch.long).to(device)
            ini_env[i] = torch.tensor([ini_env[i]], dtype=torch.long).to(device)
            if i % NUM_SEQUENCE == 0:
                his[i] = torch.tensor([[0]], dtype=torch.long).to(device)
            else:
                his[i] = torch.tensor([his[i]], dtype=torch.long).to(device)
        return ins, his, ini_env
    
    def process_raw_ws(self, ws):
        ws = ws.split(" ")
        single_ws = []
        for j in ws:
            pos_color = j.split(":")
            pos = [int(pos_color[0]) - 1]
            beaker_color = pos_color[1][:NUM_CHEMICAL_LAYERS] + "_" * (NUM_CHEMICAL_LAYERS - len(pos_color[1]))
            beaker_color_id = []
            for i in beaker_color:
                beaker_color_id.append(color_to_id[i])
            single_ws = single_ws + pos + beaker_color_id
        return torch.tensor([single_ws], dtype=torch.long)

    def replace_world_state(self, world_state):
        splitted_world_state = []
        for i in world_state:
            single_world_state = []
            for j in i:
                if j >= "0" and j <= "9":
                    single_world_state.append(int(j))
                else:
                    single_world_state.extend([color_to_id[t] for t in j])
            splitted_world_state.append(single_world_state)
        return splitted_world_state

    def process_world_state(self, world_state):
        process_world_state = []
        for i in world_state:
            ws = i.split(" ")
            single_ws = []
            for j in ws:
                pos_color = j.split(":")
                pos = [pos_color[0]]
                beaker_color = [pos_color[1] + "_" * (NUM_CHEMICAL_LAYERS - len(pos_color[1]))]
                single_ws = single_ws + pos + beaker_color
            process_world_state.append(single_ws)
        return process_world_state

    def construct_vocab(self, actions, instructions, num_filter):
        from collections import Counter
        all_instructions = [token for ins in instructions for token in ins]
        all_actions = [token for act in actions for token in act]
        instructions_counter = Counter(all_instructions)
        actions_counter = Counter(all_actions)
        filtered_instructions_tokens = [token if instructions_counter[token] >= num_filter else UNKNOWN for token in all_instructions]
        # if there is no unknown words
        filtered_instructions_tokens.append(UNKNOWN)
        instructions_set = set(filtered_instructions_tokens)
        sorted_instructions_tokens = sorted(instructions_set)
        instructions_to_id = {}
        for index, token in enumerate(sorted_instructions_tokens):
            instructions_to_id[token] = index + 1
        instructions_to_id[PAD] = 0
        sorted_instructions_tokens.append(PAD)
        self.instructions_to_id = instructions_to_id

        sorted_actions = sorted(set(actions_counter))
        actions_to_id = {}
        id_to_actions = {}
        for index, token in enumerate(sorted_actions):
            actions_to_id[token] = index + 1
            id_to_actions[index + 1] = token
        actions_to_id[PAD] = 0
        id_to_actions[0] = PAD
        sorted_actions.append(PAD)
        self.actions_to_id = actions_to_id
        self.id_to_actions = id_to_actions
        self.instruction_vocab = sorted_instructions_tokens
        self.action_vocab = sorted_actions

    def load_data(self, data_path):
        X = json.load(open(data_path))
        instructions = [[time_step["instruction"] for time_step in identifier["utterances"]] for identifier in X]
        actions = [[time_step["actions"] for time_step in identifier["utterances"]] for identifier in X]
        environments = [[time_step["after_env"] for time_step in identifier["utterances"]] for identifier in X]
        initial_environments = [[identifier["initial_env"] for i in range(NUM_SEQUENCE)] for identifier in X]

        # chain data
        environments = [j for i in environments for j in i]
        instructions = [j.split() for i in instructions for j in i]
        actions = [[START] + j + [EOS] for i in actions for j in i]
        initial_environments = [j for i in initial_environments for j in i]
        identifiers = [[i["identifier"]] * len(i["utterances"]) for i in X]
        identifiers = [id + "-" + str(j) for item in identifiers for j, id in enumerate(item)]
        history_instructions = self.construct_history_instructions(instructions)
        environments = self.construct_before_env(environments, initial_environments)
        return instructions, history_instructions, actions, initial_environments, environments, identifiers
    
    def construct_before_env(self, env, ini):
        before_env = []
        data_len = len(env)
        for i in range(data_len):
            if i % NUM_SEQUENCE == 0:
                before_env.append(ini[i])
            else:
                before_env.append(env[i-1])
        return before_env

    def construct_history_instructions(self, instructions):
        history_ins = []
        his = []
        for i in range(len(instructions)):
            if i % NUM_SEQUENCE == 0:
                his = [0]
            elif i % NUM_SEQUENCE == 1:
                his = instructions[i-1]
            else:
                his = his + instructions[i-1]
            history_ins.append(his)
        return history_ins

    def padding(self, unpadded_data):
        valid_length = [len(i) for i in unpadded_data]
        unpadded_data = [torch.tensor(i, dtype=torch.long) for i in unpadded_data]
        padded_data = pad_sequence(unpadded_data, batch_first = True)
        return padded_data, valid_length

    def replace_with_id(self, raw_data, is_instructions = True):
        if is_instructions:
            dic = self.instructions_to_id
        else:
            dic = self.actions_to_id
        
        cooked_data = [[dic[token] if token in dic.keys() else dic[UNKNOWN] for token in line] for line in raw_data]
        return cooked_data

    def construct_dataloader(self, ins, his_ins, ins_valid, his_valid, ini_env, current_env, act_id, valid_act, ground_act_id_pad, ground_act_id_pad_valid_length):
        train_data = TensorDataset(ins.to(device), his_ins.to(device), ins_valid.to(device), his_valid.to(device), ini_env.to(device), current_env.to(device), act_id.to(device), valid_act.to(device), ground_act_id_pad.to(device), ground_act_id_pad_valid_length.to(device))
        # train_data = TensorDataset(ins.to(device), his_ins.to(device), ins_valid.to(device), his_valid.to(device), ini_env.to(device), current_env.to(device), act_id.to(device), valid_act.to(device), ground_act_id_pad.to(device), ground_act_id_pad_valid_length.to(device))
        # train_sampler = RandomSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = self.batch_size)
        train_dataloader = DataLoader(train_data, batch_size = self.batch_size)
        return train_dataloader

if __name__ == "__main__":
    train = "train.json"
    dev = "dev.json"
    test = "test_leaderboard.json"
    DL = dataloader(train, dev, test)
    dl = DL.train_loader()
    for step, batch in enumerate(dl):
        batch = tuple(t.to(device) for t in batch)
        ins, his_ins, ins_valid, his_valid, ini_env, current_env, act_id, valid_act, y_true, y_true_valid = batch
        print(ini_env[:10])
        print(current_env[:10])
        break
