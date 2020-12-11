import torch
from machine_learning.modules import generate_movement_sequence, get_n_params, LSTMDiscriminator, LSTMGenerator
from machine_learning.human_data_ingestion import MAX_MOVEMENT_SEQUENCE_LENGTH, load_data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import time
# Code adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

device = "cuda"
ce_loss = nn.CrossEntropyLoss()

# this will flip the labels to the discriminator for fake instances from 0 to 1
one_sided_label_smoothing_prob = 0.2

# scale is about 4 pixels of noise on avg
normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([0.0005]))

def add_noise(discrim_input, epoch):
    if epoch > 5:
        return discrim_input
    t = normal_dist.sample((discrim_input.view(-1).size())).reshape(discrim_input.size()).to(device)
    with torch.no_grad():
        x = discrim_input + t
    return x


def validation_loss(generator, discriminator, dev_data):
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        discrim_losses = []
        generator_losses = []
        # For each batch in the dataloader
        for idx, sequence in enumerate(dev_data):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            # Format batch
            sequence = sequence.to(device)
            label = torch.LongTensor([1.0]).to(device)
            # Forward pass real batch through D
            output = discriminator(sequence)
            # Calculate loss on all-real batch
            discrim_loss_real_data = ce_loss(output, label)


            ## Train with all-fake batch
            # Generate fake sequences batch with G
            fake_sequence = generate_movement_sequence(generator, MAX_MOVEMENT_SEQUENCE_LENGTH, dev_data)
            label.fill_(0.0)
            # Classify all fake batch with D
            output = discriminator(fake_sequence.detach())
            # Calculate D's loss on the all-fake batch
            discrim_loss_fake_data = discriminator.loss(output, label)
            # Calculate the gradients for this batch
            # Add the gradients from the all-real and all-fake batches
            discrim_loss = discrim_loss_real_data + discrim_loss_fake_data
            discrim_losses.append(discrim_loss)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            label.fill_(1.0)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_sequence)
            # Calculate G's loss based on this output
            generator_loss = ce_loss(output, label)
            generator_losses.append(generator_loss)
    generator.train()
    discriminator.train()

    return sum(generator_losses) / len(generator_losses), sum(discrim_losses) / len(discrim_losses)



def train(generator, discriminator, train_data, dev_data, test_data, num_epochs, discrim_lr=0.0001, generator_lr=0.0001):
    start = time.time()
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discrim_lr)
    generator_optimizer = optim.Adam(generator.parameters(), lr=generator_lr)

    # Training Loop
    generator.train()
    discriminator.train()

    # Lists to keep track of progress
    img_list = []
    G_train_losses = []
    D_train_losses = []

    G_eval_losses = []
    D_eval_losses = []
    try:
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            g_loss_avg = 0.0
            d_loss_avg = 0.0
            print(f"Epoch {epoch}; {len(train_data)} examples")
            D_x = 0.0
            D_G_z1 = 0.0
            D_G_z2 = 0.0
            for idx, sequence in enumerate(train_data):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                discriminator.zero_grad()
                # Format batch
                sequence = sequence.to(device)
                label = torch.LongTensor([1.0]).to(device)
                # Forward pass real batch through D
                sequence = add_noise(sequence, epoch)
                output = discriminator(sequence)
                # Calculate loss on all-real batch
                discrim_loss_real_data = ce_loss(output, label)
                # Calculate gradients for D in backward pass
                discrim_loss_real_data.backward()
                D_x += output[0][1].item()

                ## Train with all-fake batch
                # Generate fake sequences batch with G
                fake_sequence = generate_movement_sequence(generator, MAX_MOVEMENT_SEQUENCE_LENGTH, train_data)
                if random.random() < one_sided_label_smoothing_prob / (epoch + 1):
                    label.fill_(1.0)
                else:
                    label.fill_(0.0)
                # Classify all fake batch with D
                # Detach it because we don't want to back propagate to the generator yet
                output = discriminator(add_noise(fake_sequence, epoch).detach())
                # Calculate D's loss on the all-fake batch
                discrim_loss_fake_data = discriminator.loss(output, label)
                # Calculate the gradients for this batch
                discrim_loss_fake_data.backward()
                D_G_z1 += output[0][1].item()
                # Add the gradients from the all-real and all-fake batches
                discrim_loss = discrim_loss_real_data + discrim_loss_fake_data
                d_loss_avg += discrim_loss.item()
                # Update D
                discriminator_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                label.fill_(1.0)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(fake_sequence)
                # Calculate G's loss based on this output
                generator_loss = ce_loss(output, label)
                g_loss_avg += generator_loss.item()
                # Calculate gradients for G
                generator_loss.backward()
                D_G_z2 += output[0][1].item()
                # Update G
                generator_optimizer.step()

                # Output training stats
                if idx % 1000 == 0:

                    print('[%d minutes][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % ((time.time() - start ) / 60000, epoch, num_epochs, idx, len(train_data),
                             discrim_loss.item(), generator_loss.item(), D_x / 1000, D_G_z1 / 1000, D_G_z2 / 1000))
                    D_x = 0
                    D_G_z1 = 0
                    D_G_z2 = 0

                # Save Losses for plotting later


            g_loss_avg /= len(train_data)
            d_loss_avg /= len(train_data)
            G_train_losses.append(g_loss_avg)
            D_train_losses.append(d_loss_avg)

            gl, dl = validation_loss(generator, discriminator, dev_data)
            G_eval_losses.append(g_loss_avg)
            D_eval_losses.append(d_loss_avg)

            print(f"Train generator loss {g_loss_avg}; Train discriminator loss {d_loss_avg}; Eval generator_loss {gl}; Eval discriminator loss {dl}")

            print("Saving models")
            loss1_str = "{0:.5g}".format(gl)
            loss2_str = "{0:.5g}".format(dl)

            torch.save(generator.state_dict(), f"machine_learning/models/gan_generator_{epoch}_{loss1_str}_{loss2_str}.mdl")
            torch.save(generator.state_dict(), f"machine_learning/models/gan_discriminator_{epoch}_{loss1_str}_{loss2_str}.mdl")
    except KeyboardInterrupt:
        plt.figure(figsize=(10, 5))
        plt.title("Generator pretraining losses")
        plt.plot(G_train_losses, label="Generator train")
        plt.plot(D_train_losses, label="Discriminator train")
        plt.plot(G_eval_losses, label="Generator eval")
        plt.plot(D_eval_losses, label="Discriminator eval")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig(f"machine_learning/images/gan_losses.png")

gen_losses = [0.313, 1.15, 0.64, 1.31, 0.31, 1.31, 1.31, 1.31, 1.31, 1.31]
discrim_losses = [1.63, 1.18, 1.35, .63, 1.63, 0.63, 0.63, 0.63, 0.63, 0.63]

plt.figure(figsize=(10, 5))
plt.title("Generator pretraining losses")
plt.plot(gen_losses, label="Generator train")
plt.plot(discrim_losses, label="Discriminator train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig(f"machine_learning/images/gan_losses.png")

if __name__ == "__main__":
    lstm = LSTMGenerator()
    print(get_n_params(lstm))
    lstm = lstm.to(device)
    raw_mvmts_train, raw_mvmts_dev, raw_mvmts_test = load_data()

    lstm.load_state_dict(torch.load(
        "machine_learning/models/pretrained_generator_55_0.00017816_{'input_size': 3, 'hidden_size': 96, 'dropout': 0.2, 'bidrectional': False, 'num_layers': 4}_prioritize_xy.mdl"))

    discrim = LSTMDiscriminator()
    discrim = discrim.to(device)
    print(get_n_params(discrim))

    train(lstm, discrim, raw_mvmts_train, raw_mvmts_dev, raw_mvmts_test, 10)


