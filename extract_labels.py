""" Extracts interaction-level and instruction-level labels for the given file.

Note that train.json and dev.json contain both input AND output gold labels; however,
we use a CSV to compare your predictions to the gold labels. So gold labels for train/dev 
are in two files: train.json/dev.json, and train_*_y.csv/dev_*_y.csv. You can use either
the JSON or CSVs for training; you should use the CSVs when running evaluate.py.
"""
import json
import sys

examples = json.loads(open(sys.argv[1]).read())

instruction_outfile = open(sys.argv[2], "w")
interaction_outfile = open(sys.argv[3], "w")

interaction_outfile.write("id,final_world_state\n")
instruction_outfile.write("id,final_world_state\n")

for example in examples:
    identifier = example["identifier"]
    final_env = example["utterances"][-1]["after_env"]
    interaction_outfile.write(identifier + "," + final_env + "\n")

    for i, instruction in enumerate(example["utterances"]):
        instruction_outfile.write(identifier + "-" + str(i) + "," + instruction["after_env"] + "\n")

interaction_outfile.close()
instruction_outfile.close()
