from machine_learning.modules import LSTMGenerator, generate_movement_sequence, load_data, device
from machine_learning.human_data_ingestion import MAX_MOVEMENT_SEQUENCE_LENGTH
import torch

def compare_generators(paths, data):
    lstms = {path: LSTMGenerator() for path in paths}

    for path in lstms:
        lstm = lstms[path].to(device)
        lstm.load_state_dict(torch.load(path))
        lstm.eval()
        print(f"Path {path}")
        fake_sequence = generate_movement_sequence(lstm, MAX_MOVEMENT_SEQUENCE_LENGTH, data)
        print(fake_sequence)

if __name__ == "__main__":
    raw_mvmts_train, raw_mvmts_dev, raw_mvmts_test = load_data()
    best_generators = ["machine_learning/models/pretrained_generator_59_0.0013473_{'input_size': 3, 'hidden_size': 96, 'dropout': 0.2, 'bidrectional': False, 'num_layers': 4}_prioritize_action.mdl",
                       "machine_learning/models/pretrained_generator_55_0.00017816_{'input_size': 3, 'hidden_size': 96, 'dropout': 0.2, 'bidrectional': False, 'num_layers': 4}_prioritize_xy.mdl",
                       "machine_learning/models/gan_generator_7_0.31326_1.6265.mdl",
                       "machine_learning/models/gan_generator_6_0.31326_1.6265.mdl"]
    compare_generators(best_generators, raw_mvmts_dev)