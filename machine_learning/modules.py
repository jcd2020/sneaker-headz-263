import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from machine_learning.human_data_ingestion import load_data, MAX_MOVEMENT_SEQUENCE_LENGTH, actions
import random
import matplotlib.pyplot as plt

# code adapted from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

LSTM_CONFIG = {"input_size": 3, "hidden_size": 96, "dropout": 0.2, "bidrectional": False, "num_layers": 4}
CLFR_CONFIG = {"input_size": 3, "hidden_size": 64, "dropout": 0.5, "bidrectional": True, "num_layers": 3}
device = "cuda"

class LSTMGenerator(nn.Module):
    def __init__(self):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = LSTM_CONFIG["hidden_size"]
        self.input_dim = LSTM_CONFIG["input_size"]
        self.num_layers = LSTM_CONFIG["num_layers"]

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(LSTM_CONFIG["input_size"], LSTM_CONFIG["hidden_size"], dropout=LSTM_CONFIG["dropout"],
                            bidirectional=LSTM_CONFIG["bidrectional"], num_layers=LSTM_CONFIG["num_layers"])

        # The linear layer that maps from hidden state space to tag space
        # factor of 2 bc bidirectional LSTM
        self.multiplier = 2 if LSTM_CONFIG["bidrectional"] else 1

        self.coordinates = nn.Sequential(nn.Linear(self.multiplier*LSTM_CONFIG["hidden_size"], int(self.multiplier*LSTM_CONFIG["hidden_size"]/2)),
                                         nn.ReLU(), nn.Linear(int(self.multiplier*LSTM_CONFIG["hidden_size"]/2), 2))
        self.button_action = nn.Sequential(nn.Linear(self.multiplier*LSTM_CONFIG["hidden_size"], int(self.multiplier*LSTM_CONFIG["hidden_size"]/2)),
                                           nn.ReLU(), nn.Linear(int(self.multiplier*LSTM_CONFIG["hidden_size"]/2), len(actions)))

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        xy = self.coordinates(lstm_out.view(len(sequence), -1))
        button = self.button_action(lstm_out.view(len(sequence), -1))

        return (xy, button), lstm_out[-1]

    def loss(self, target, output):
        (xy, button) = output

        target_xy = target.permute(1, 0)[:2].permute(1, 0)
        xy_loss = torch.nn.functional.mse_loss(xy, target_xy)

        target_action = target.permute(1, 0)[2]
        action_loss = nn.CrossEntropyLoss()(button, target_action.type(torch.LongTensor).to(device))

        # hard coded constants that make each element approx equal size
        return (100 * xy_loss + action_loss), xy_loss, action_loss

    def init_state(self, batch_sz=1):
        return (torch.zeros(self.num_layers*self.multiplier, batch_sz, self.hidden_dim),
                torch.zeros(self.num_layers*self.multiplier, batch_sz, self.hidden_dim))

    def generate(self, input, prev_state):
        output, state = self.lstm(input, prev_state)

        xy = self.coordinates(output.view(len(output), -1))
        button = self.button_action(output.view(len(output), -1))
        chosen_action = torch.distributions.Categorical(torch.nn.functional.softmax(button)).sample()
        return xy, chosen_action, state


class LSTMDiscriminator(nn.Module):
    def __init__(self):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_dim = CLFR_CONFIG["hidden_size"]

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(CLFR_CONFIG["input_size"], CLFR_CONFIG["hidden_size"], dropout=CLFR_CONFIG["dropout"], bidirectional=CLFR_CONFIG["bidrectional"], num_layers=CLFR_CONFIG["num_layers"])

        # The linear layer that maps from hidden state space to tag space
        # factor of 2 bc bidirectional LSTM
        self.fc = nn.Sequential(nn.Linear(2*CLFR_CONFIG["hidden_size"], CLFR_CONFIG["hidden_size"]), nn.ReLU(), nn.Linear(CLFR_CONFIG["hidden_size"], 2))

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        preds = torch.nn.functional.softmax(self.fc(lstm_out[-1]))
        return preds

    def loss(self, scores, target):
        action_loss = nn.CrossEntropyLoss()(scores, target.type(torch.LongTensor).to(device))
        return action_loss




def generate_movement_sequence(model, seq_length, raw_mvmts_train, initial_sequence_size=25):
    initial_mvmts = random.choice(raw_mvmts_train)[0:initial_sequence_size]
    return generate_raw_mvmt_sequence_from_init(model, seq_length, initial_mvmts).to(device)

def generate_raw_mvmt_sequence_from_init(model, seq_length, random_start_mvmts):
    random_start_mvmts = random_start_mvmts.to(device)

    mvmts = [mvmt[0] for mvmt in random_start_mvmts]


    state_h, state_c = model.init_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    # update hiddne states based on feeding in starting sequence
    _, _, (state_h, state_c) = model.generate(random_start_mvmts, (state_h, state_c))

    previous_movement = random_start_mvmts[-1]
    for i in range(0, seq_length - len(mvmts)):
        xy, button, (state_h, state_c) = model.generate(previous_movement.view(1,1,-1), (state_h, state_c))
        next_mvmt = torch.cat([xy[0], button], dim=0)
        previous_movement = next_mvmt
        mvmts.append(next_mvmt)

    return torch.stack(mvmts).unsqueeze(1)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def pretrain_generator(generator, data, epochs=300, lr=0.0001):
    generator.train()
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    split = int(0.9*len(data))
    data, validation_data = data[:split], data[split:]

    eval_losses = []
    train_losses = []

    def validation_loss():
        with torch.no_grad():
            generator.eval()
            valid_loss = 0.0
            for seq in validation_data:
                seq = seq.to(device)
                output, _ = generator(seq)
                loss, _, _ = generator.loss(seq.view(len(seq), -1), output)
                valid_loss += loss.item()

            generator.train()
        return valid_loss / len(validation_data)

    min_eval_loss = float('inf')
    for epoch in range(epochs):
        avg_loss = 0.0
        avg_xy_loss = 0.0
        avg_action_loss = 0.0
        print(f"Epoch {epoch}; {len(data)} sequence items")
        random.shuffle(data)
        for idx, raw_mvmt_seq in enumerate(data):
            raw_mvmt_seq = raw_mvmt_seq.to(device)
            optimizer.zero_grad()
            output, _ = generator(raw_mvmt_seq)
            loss, xy_loss, action_loss = generator.loss(raw_mvmt_seq.view(len(raw_mvmt_seq), -1), output)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_xy_loss += xy_loss.item()
            avg_action_loss += action_loss.item()
        eval_loss = validation_loss()
        print(f"Train loss {avg_loss / len(data)}, {avg_xy_loss / len(data)}, {avg_action_loss / len(data)}; Eval loss {eval_loss}")
        eval_losses.append(eval_loss)
        train_losses.append(avg_loss / len(data))
        if eval_loss < min_eval_loss:
            print(f"Saving model")
            loss_str = "{0:.5g}".format(eval_loss)
            torch.save(generator.state_dict(), f"machine_learning/models/trained_generator_{epoch}_{loss_str}_{str(LSTM_CONFIG)}.mdl")
            min_eval_loss = eval_loss

    plt.figure(figsize=(10, 5))
    plt.title("Generator pretraining losses")
    plt.plot(eval_losses, label="Eval")
    plt.plot(train_losses, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(f"machine_learning/images/pretraining_losses.png")
    plt.clf()

if __name__ == "__main__":
    lstm = LSTMGenerator()
    print(get_n_params(lstm))
    lstm = lstm.to(device)
    raw_mvmts_train, raw_mvmts_dev, raw_mvmts_test = load_data()
    pretrain_generator(lstm, raw_mvmts_train, epochs=60)