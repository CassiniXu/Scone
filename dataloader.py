import numpy as np
import json
from fsa import EOS, ACTION_SEP, NO_ARG
import string
UNKNOWN = "<UNKNOWN>"
PAD = "<PAD>"
START = "<START>"
NUM_SEQUENCE = 5
NUM_CHEMICAL_LAYERS = 4

color_to_id = {"_": 0, "y": 1, "o": 2, "g": 3, "r": 4, "b": 5, "p": 6}
class dataloader():
    def __init__(self, train, dev, test, batch_size = 32, num_filter=10):
        import torch
        self.instructions_to_id = None
        self.actions_to_id = None
        self.id_to_actions = None
        self.train = train
        self.dev = dev
        self.test = test
        self.batch_size = 32
        instructions, his_instructions, actions, initial_environments, environments, identifiers = self.load_data(self.train)
        self.construct_vocab(actions, instructions, num_filter)
        dev_ins, dev_his, _, dev_ini_env, _ = self.load_data(self.dev)
        test_ins, test_his, _, test_ini_env, _ = self.load_data(self.test)

        """
        prepare for training data
        """
        # replace with id
        train_id = self.replace_with_id(instructions)
        train_his_id = self.replace_with_id(his_instructions)
        act_id = self.replace_with_id(actions, False)

        # padding
        train_id_pad, train_valid_length = self.padding(train_id)
        train_his_id_pad, train_his_valid_length = self.padding(train_his_id)
        act_id_pad, act_valid_length = self.padding(act_id)

        # to tensor
        train_id_pad = torch.tensor(train_id_pad,  dtype=torch.long)
        train_his_id_pad = torch.tensor(train_his_id_pad,  dtype=torch.long)
        train_valid_length = torch.tensor(train_valid_length, dtype=torch.long)
        train_his_valid_length = torch.tensor(train_his_valid_length, dtype=torch.long)
        act_id_pad = torch.tensor(act_id_pad, dtype=torch.long)
        act_valid_length = torch.tensor(act_valid_length, dtype=torch.long)

        # process and replace world state
        initial_environments = self.process_world_state(initial_environments)
        environments = self.process_world_state(environments)
        initial_environments = self.replace_world_state(initial_environments)
        environments = self.replace_world_state(environments)
        self.train_dataloader = self.construct_dataloader(train_id_pad, train_his_id_pad, train_valid_length, train_his_valid_length, initial_environments, environments, act_id_pad, act_valid_length)
        


    def replace_world_state(self, world_state):
        splitted_world_state = []
        for i in world_state:
            single_world_state = []
            for j in i:
                if j >= "0" and j <= "9":
                    single_world_state.append(int(j))
                else:
                    single_world_state.append(color_to_id[j])
            splitted_world_state.append(single_world_state)
        return splitted_world_state

    def process_world_state(self, world_state):
        process_world_state = []
        for i in world_state:
            ws = i.split(" ")
            single_ws = []
            for j in ws:
                pos_color = j.split(":")
                pos = pos_color[0]
                beaker_color = pos_color[1:] + ["_" * (4 - len(pos_color[1:]))]
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
        for index, token in enumerate(sorted):
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
        instructions = [j[0].split() for i in instructions for j in i]
        actions = [[START] + j + [EOS] for i in actions for j in i]
        initial_environments = [j for i in initial_environments for j in i]
        identifiers = [[i["identifier"]] * len(i["utterances"]) for i in X]
        identifiers = [id + "-" + str(j) for item in identifiers for j, id in enumerate(item)]
        history_instructions = self.construct_history_instructions(instructions)
        return instructions, his_instructions, actions, initial_environments, environments, identifiers
    
    def construct_history_instructions(self, instructions):
        history_ins = []
        his = []
        for i in instructions:
            if i % NUM_SEQUENCE == 0:
                his = []
            else:
                his = his + instructions[i-1]
            history_ins.append(his)
        return history_ins

    def padding(self, unpadded_data):
        valid_length = [len(i) for i in unpadded_data]
        from torch.nn.utils.rnn import pad_sequence
        padded_data = pad_sequence(unpadded_data, batch_first = True)
        return padded_data, valid_length

    def replace_with_id(self, raw_data, is_instructions = True):
        if is_instructions:
            dic = self.self.instructions_to_id
        else:
            dic = self.actions_to_id
        
        cooked_data = [[dic[token] if token in dic.keys() else dic[UNKNOWN] for token in line] for line in raw_data]
        return cooked_data
    
    def construct_dataloader(self, ins, his_ins, ins_valid, his_invalid, ini_env, current_env, act_id, valid_act):
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        train_data = TensorDataset(ins, his_ins, ins_valid, his_invalid, ini_env, current_env, act_id, valid_act)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = self.batch_size)
        return train_dataloader



