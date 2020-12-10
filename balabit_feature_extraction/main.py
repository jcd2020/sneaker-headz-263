import os

import balabit_feature_extraction.settings as st
import balabit_feature_extraction.rawdata2actions as rd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle as pkl
import random

# SESSION_CUT = 2
# modeling unit: ACTION
def main_ACTION():
    st.SESSION_CUT = 2
    print("***Computing training features")
    st.CASE = 'training'
    action_sets, raw_movements = rd.process_files(st.CASE)
    random.shuffle(raw_movements)
    train_mvmts = raw_movements[:int(0.8 * len(raw_movements))]
    validation_mvmts = raw_movements[int(0.8 * len(raw_movements)):]

    with open("machine_learning/data/human_train_sessions_raw_mvmts.pkl", "wb+") as f:
        pkl.dump(train_mvmts, f)

    with open("machine_learning/data/human_validation_sessions_raw_mvmts.pkl", "wb+") as f:
        pkl.dump(validation_mvmts, f)

    print('***Evaluating on the test set')
    st.CASE = 'test'
    action_sets, raw_movements = rd.process_files(st.CASE)

    with open("machine_learning/data/human_test_sessions_raw_mvmts.pkl", "wb+") as f:
        pkl.dump(raw_movements, f)
    rd.process_files(st.CASE)
    return


main_ACTION()
