import pickle as pkl
import os
import torch
os.chdir("/home/jeremydohmann/Documents/hdd/home/Documents/Courses2020/CS263/Projects/Final project/sneaker-headz-263")

# define sequence lengths for RNN
MAX_ACTION_SEQUENCE_LENGTH = 20
MAX_MOVEMENT_SEQUENCE_LENGTH = 200

# define action space discrete mappings for RNN
actions = [('Move', 'NoButton'),
              ('Drag', 'NoButton'),
              ('Pressed', 'Left'),
              ('Pressed', 'Right'),
              ('Released', 'Right'),
              ('Released', 'XButton'),
              ('Pressed', 'XButton'),
              ('Released', 'Left')]
action2idx = {action: idx for idx, action in enumerate(actions)}
idx2action = {v: k for k,v in action2idx.items()}

def segment_instances(data, max_seq_length):
    # convert raw session data into fixed length sequences for use by RNN
    avg_timestep = 0
    counter = 1
    for session in data:
        prev_time = float(session[0]["client timestamp"])
        for step in session[1:]:
            time = float(step["client timestamp"])
            avg_timestep += time - prev_time
            prev_time = time
            counter += 1
    print(f"Average duration of timestep {avg_timestep / counter}")

    sequences = []
    for session in data:
        for i in range(0, len(session), max_seq_length):
            chunk = session[i:i+max_seq_length]
            sequences.append(chunk)
    return sequences

def discretize_raw_data(raw_mvmts):
    def discretize(row):
        return [
                row["x"] / 4000,
                row["y"] / 4000,
                action2idx[(row["state"], row["button"])]
         ]


    sessions = [[discretize(row) for row in session] for session in raw_mvmts]

    for session in sessions:
        for i, row in enumerate(session):
            if row[0] > 1 or row[1] > 1:
                row[0] = session[i-1][0] if i > 0 else 1
                row[1] = session[i-1][1] if i > 0 else 1

    return sessions

def torchify_segments(segment_data):
    res = []
    for segment in segment_data:
        res.append(torch.unsqueeze(torch.Tensor(segment), 1))
    return res

def load_data():
    # with open("machine_learning/data/human_train_sessions_action_sets.pkl", "rb") as f:
    #     actions_train_sequences = segment_instances(pkl.load(f), MAX_ACTION_SEQUENCE_LENGTH)
    with open("machine_learning/data/human_train_sessions_raw_mvmts.pkl", "rb") as f:
        raw_mvmts_train = discretize_raw_data(segment_instances(pkl.load(f), MAX_MOVEMENT_SEQUENCE_LENGTH))

    with open("machine_learning/data/human_validation_sessions_raw_mvmts.pkl", "rb") as f:
        raw_mvmts_dev = discretize_raw_data(segment_instances(pkl.load(f), MAX_MOVEMENT_SEQUENCE_LENGTH))
    # with open("machine_learning/data/human_test_sessions_action_sets.pkl", "rb") as f:
    #     actions_test_sequences = segment_instances(pkl.load(f), MAX_ACTION_SEQUENCE_LENGTH)
    with open("machine_learning/data/human_test_sessions_raw_mvmts.pkl", "rb") as f:
        raw_mvmts_test = discretize_raw_data(segment_instances(pkl.load(f), MAX_MOVEMENT_SEQUENCE_LENGTH))

    return torchify_segments(raw_mvmts_train), \
           torchify_segments(raw_mvmts_dev), \
           torchify_segments(raw_mvmts_test)

if __name__ == "__main__":
    load_data()