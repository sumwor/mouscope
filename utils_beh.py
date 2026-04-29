# utility functions for behaviral analysis
import csv
import numpy as np

# Deeplabcut related, and MotionSequence related functions

def load_DLC(filepath):
    # load DLC results
    data = {}
    nFrames = 0

    if isinstance(filepath, str):
        with open(filepath) as csv_file:
            #print("Loading data from: " + filePath)
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # scorer
                    data[row[0]] = row[1]
                    line_count += 1
                elif line_count == 1:  # body parts
                    bodyPartList = []
                    for bb in range(len(row) - 1):
                        if row[bb + 1] not in bodyPartList:
                            bodyPartList.append(row[bb + 1])
                    data[row[0]] = bodyPartList
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 2:  # coords
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 3:  # actual coords
                    # print({", ".join(row)})
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        if np.mod(ii, 3) == 0:
                            data[body] = {}
                        data[body][tempList[np.mod(ii, 3)]] = [float(row[ii + 1])]
                    #self.t.append(0)
                    line_count += 1
                    nFrames += 1

                else:
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                    #self.t.append(self.nFrames*(1/self.fps))
                    line_count += 1
                    nFrames += 1

            print(f'Processed {line_count} lines.')
    
    return data