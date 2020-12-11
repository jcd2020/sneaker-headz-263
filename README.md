# sneaker-headz-263

Use either conda, pipenv, or venv to create a virutal environment with the dependencies in `requirements.txt`.

We recommend importing the project into PyCharm to have the path names and environment properly configured. In order to extract the Balabit data to the necessary format, run `balabit_feature_extraction/main.py.` The code in this directory was taken from https://arxiv.org/abs/1810.04668.

The original Balabit data, downloaded from https://ms.sapientia.ro/~manyi/mousedynamics/#dataset, is in `user_data`.

In order to train an LSTM run `machine_learning/modules.py`. In order to train a GAN run `machine_learning/lstm_gan.py`. Configure the device to meet your needs (i.e. if you don't have a gpu, set it to "cpu").

Visualizations of the trajectories are in `pngs` and were produced using `plot.py`.

Histograms of the trajectory statistics were produced using `machine_learning/evaluating_results.py` and are saved in `machine_learning/images/Histograms`.

Some code to run bots is in `bot`.

