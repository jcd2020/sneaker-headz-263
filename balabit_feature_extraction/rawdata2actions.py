import os
import csv
import pandas as pd

import balabit_feature_extraction.settings as st
import balabit_feature_extraction.actions as actions


# 2017.09.07, Lyon
# new split, splits the raw data into mouse actions
# {MM}*DD
# {MM}*PC
def processSession1(rows):
    # Opens a session file containing raw mouse events and creates a file segmented into actions
    # "CSV file structure: record timestamp, client timestamp, button, state, x, y "

    # line counter needed for the n_from and n_to fields
    # rows belonging to a segmented action [n_from, n_to]
    counter = 1
    data = []
    action_sequence = []
    prevrow = None
    n_from = 2
    n_to = 2
    for row in rows:
            counter = counter + 1
            # Skip duplicates
            if prevrow != None and prevrow == row:
                continue
            # Skip equal timestamps
            # if prevrow != None and row['client timestamp'] == prevrow['client timestamp']:
            #     continue

            item = {
                "x": row['x'],
                "y": row['y'],
                "t": row['client timestamp'],
                "button": row['button'],
                "state": row['state']
            }
            # SCROLLs are not actions
            # therefore are ignored
            if row["button"] == 'Scroll':
                if prevrow != None:
                    item['x'] = prevrow['x']
                    item['y'] = prevrow['y']
                # continue
            if row['button'] == 'Left' and row['state'] == 'Released':
                # n_to = counter
                # print("Left - Released: "+str(n_from)+"-"+str(n_to))
            # ha a Right clicket is PC-nek minositjuk!!!! ????
            # if row['state'] == 'Released':
                data.append(item)
                # is it a short sequence?
                if len(data) <= 2:
                    # print(str(n_from)+"--"+str(counter ))
                    data = []
                    n_from = counter
                    continue

                # A Drag Drop Action (4) ends here.
                # It can be a compound action: {MM}*DD - several MM actions followed by a DD action
                if prevrow != None and prevrow['state'] == 'Drag':
                    # if actions.GLOBAL_DEBUG:
                    #     print(str(counter))
                    #     print(item)
                    n_to =counter
                    action_sequence = actions.processDragActions(data, action_sequence, n_from, n_to)

                # A Point Click Action (3) ends here.
                # It can be a compunded action: {MM}*PC - several MM actions followed by a DD action
                if prevrow != None and prevrow['state'] == 'Pressed':
                    # if actions.GLOBAL_DEBUG:
                    #     print(str(counter))
                    #     print(item)
                    n_to = counter
                    action_sequence = actions.processPointClickActions(data, action_sequence, n_from, n_to)

                # It starts a new action
                data = []
                n_from = n_to +1
            else:
                if int(item['x'])<st.X_LIMIT and int(item['y']) <st.Y_LIMIT:
                    data.append(item)
            prevrow = row
    n_to = counter
    action_sequence = actions.processPointClickActions(data, action_sequence,n_from, n_to)
    return action_sequence







# input: case {'training','test'}
# output: output/balabit_featutes_training.csv OR output/balabit_featutes_test.csv
def process_files ( case ):
    dlabels = {}
    action_sets = []
    raw_movements = []

    if case == 'test':
        directory = os.fsencode(st.BASE_FOLDER + st.TEST_FOLDER)
    else:
        directory = os.fsencode(st.BASE_FOLDER + st.TRAINING_FOLDER)


    counter = 0
    for fdir in os.listdir(directory):
        dirname = os.fsdecode(fdir)
        print('User: ' + dirname)
        if case == 'test':
            userdirectory = st.BASE_FOLDER + st.TEST_FOLDER + '/' + dirname
        else:
            userdirectory = st.BASE_FOLDER + st.TRAINING_FOLDER + '/' + dirname
        # is_legal is not used in case of training
        is_legal = 0
        userid = dirname[4:len(dirname)]
        for file in os.listdir(userdirectory):
            fname = os.fsdecode(file)
            filename = userdirectory + '/' + os.fsdecode(file)
            sessionid = str(fname[8:len(fname)])
            counter += 1
            # nem minden teszfajlnak ismert a cimkeje
            if case == 'test' and not sessionid in dlabels:
                continue
            print('File: ' + fname)
            if case == 'test':
                is_legal = dlabels[sessionid]

            with open(filename) as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                rows = [{k: int(v) if v.isnumeric() else v for (k,v) in row.items()} for row in rows]
                action_sequence = processSession1(rows)
                raw_movements.append([row for row in rows if row["button"] != "Scroll"])

            # end split
            action_sequence = [x for x in action_sequence if x is not None]

            action_sets.append(action_sequence)

    print("Num session files: " + str(counter))

    print( case )
    if case == 'test':
        print("public labels: " + str(len(dlabels)))
    print("SESSION_CUT: " + str(st.SESSION_CUT))
    if st.SESSION_CUT == 1:
        print("NUM_ACTIONS: "+str(st.NUM_ACTIONS))

    return action_sets, raw_movements
