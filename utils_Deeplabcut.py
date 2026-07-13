import ast
import csv
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from tqdm import tqdm

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
plt.ion()
import imageio
from skimage import color

from pyPlotHW import StartPlots
from utils_beh import butter_lowpass_filter, distance_points_to_line, read_video


class DLCSession:

    def __init__(self, filePath, videoPath, rodspeedPath,analysisPath, fps):
        """

        :param filePath: DLC csv path
        :param videoPath:
        :param rodspeedPath: rod speed csv path
        :param analysisPath:
        :param fps: number (frames per second); path: file path with time stamps
        """

        self.filePath = filePath
        self.videoPath = videoPath
        self.rodPath = rodspeedPath
        self.nFrames = 0
        # test if fps is a number or a file path
        if isinstance(fps, (int, float)):
            self.fps = fps
        else:
            # load the timeStamp csv
            time_raw = pd.read_csv(fps, header=None)
            self.t = np.array(time_raw[0]-time_raw[0][0])/1000
            self.t_start = time_raw[0][0]
        # read data
        self.data = self.read_data()
        self.analysis = analysisPath
        self.fieldSize = 40 # in centimeter, used to convert px to cm
        if not os.path.exists(self.analysis):
            os.makedirs(self.analysis)
        #self.video = self.read_video()

    def get_confidence(self, p_threshold, savefigpath):
        # get potentially outlier frames by confidence
        # focus on body parts that are most likely to be wrong:
        # tail 1, nose, left/right hand/foot
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'spine 1', 'spine 2', 'spine 3', 'tail 1']
        outliers = []
        for f in range(self.nFrames):
            for bp in body_parts:
                if self.data[bp]['p'][f]<p_threshold:
                    outliers.append(f)
                break

        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
        #self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath,figName))
            plt.close()

    def get_jump(self, px_threshold, savefigpath):
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'tail 1', 'tail 2', 'tail 3']
        outliers = []
        for f in range(self.nFrames-1):
            for bp in body_parts:
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                if np.sqrt(dx2+dy2) > px_threshold:
                    outliers.append(f+1)
                break


        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
            # self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath, figName))
            plt.close()

    def kp_jump_dist(self):
        # calculate cross frame keypoint jumps and plot the distrubution
        body_parts = self.data['bodyparts']
        kp_jumps = {}
        for f in range(self.nFrames-1):
            for bp in body_parts:
                if f==0:
                    kp_jumps[bp] = []
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                kp_jumps[bp].append(np.sqrt(dx2+dy2))

        bins = np.arange(0.0,30,0.2)
        fig, ax = plt.subplots(3,5,sharey=True)
        for idx, bp in enumerate(body_parts):
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].hist(kp_jumps[bp], bins= bins)
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].set_xlabel(bp)

    def moving_trace(self, savefigpath):
        """ plot animal moving trace in the field"""
        if not hasattr(self, 'arena'):
            savedatapath = os.path.join(savefigpath, 'arena_coordinates.csv')
            if not os.path.exists(savedatapath):
                self.arena = frame_input(self.videoPath)
                # save the results:
                with open(savedatapath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['upper left',
                                     'upper right',
                                     'lower right',
                                     'lower left'])
                    writer.writerow([self.arena['upper left'],
                                    self.arena['upper right'],
                                    self.arena['lower right'],
                                    self.arena['lower left']])
                    f.close()
            else:
                # read data from file
                tempdata= pd.read_csv(savedatapath)
                self.arena = {}
                for key in tempdata.keys():
                    self.arena[key] = ast.literal_eval(tempdata[key].values[0])

                # convert px to cm
                # calculate the length of each side in pixels, get the average, then convert to cm
        sideLength = []
        arenaKeys = list(self.arena.keys())
        for kidx in range(len(arenaKeys)):
            key1 = arenaKeys[kidx]
            if kidx < len(arenaKeys)-1:
                key2 = arenaKeys[kidx + 1]
            else:
                key2 = arenaKeys[0]
            sideLength.append(np.sqrt((self.arena[key1][0]-self.arena[key2][0])**2+
                                              (self.arena[key1][1]-self.arena[key2][1])**2))
        # save the coordinates in analysis folder
        self.px2cm = self.fieldSize/np.mean(sideLength)

        arena_x = [self.arena['upper left'][0], self.arena['upper right'][0],
                   self.arena['lower right'][0], self.arena['lower left'][0], self.arena['upper left'][0]]
        arena_y = [self.arena['upper left'][1], self.arena['upper right'][1],
                   self.arena['lower right'][1], self.arena['lower left'][1], self.arena['upper left'][1]]

        # get the instantaneous distance from center
        slope1 = (self.arena['upper left'][1] - self.arena['lower right'][1]) / (self.arena['upper left'][0] - self.arena['lower right'][0])
        slope2 = (self.arena['lower left'][1] - self.arena['upper right'][1]) / (self.arena['lower left'][0] - self.arena['upper right'][0])

        # Calculate the x-coordinate of the intersection point
        x_intersection = ((self.arena['upper right'][1] - self.arena['upper left'][1]) + slope1 * self.arena['upper left'][0] - slope2 * self.arena['upper right'][0]) / (slope1 - slope2)

        # Calculate the y-coordinate of the intersection point
        y_intersection = slope1 * (x_intersection - self.arena['upper left'][0]) + self.arena['upper left'][1]

        self.center_point = [x_intersection, y_intersection]

        self.dist_center = np.sqrt((np.array(self.data['tail 1']['x'])-self.center_point[0])**2 +
                                   (np.array(self.data['tail 1']['y'])-self.center_point[1])**2)

        if hasattr(self, 'px2cm'):
            self.dist_center = self.dist_center*self.px2cm

        tracePlot = StartPlots()
        tracePlot.ax.plot(arena_x, arena_y)
        tracePlot.ax.plot(self.data['tail 1']['x'], self.data['tail 1']['y'])
        tracePlot.ax.axis('equal')
        # Hide the x and y axes
        tracePlot.ax.axis('off')
        tracePlot.save_plot('Moving trace.tif', 'tif', savefigpath)
        tracePlot.save_plot('Moving trace.svg', 'svg', savefigpath)

    def get_time_in_center(self):
        """calculate time spent in the center"""
        if hasattr(self, 'arena'):
            # do the calculation
            side_length = np.sqrt((self.arena['upper left'][0] - self.arena['upper right'][0])**2 + (self.arena['upper left'][1] - self.arena['upper right'][1])**2)
            self.center = {}
            self.center['upper left'] = (self.arena['upper left'][0] + side_length/4,
                                         self.arena['upper left'][1] + side_length/4)
            self.center['upper right'] = (self.arena['upper right'][0] - side_length/4,
                                          self.arena['upper right'][1] + side_length / 4)
            self.center['lower right'] = (self.arena['lower right'][0] - side_length/4,
                                             self.arena['lower right'][1] - side_length / 4)
            self.center['lower left'] = (self.arena['lower left'][0] + side_length/4,
                                            self.arena['lower right'][1] - side_length / 4)

            # determine if tail 1 is inthe center area\
            x_left = (self.center['upper left'][0] + self.center['lower left'][0])/2
            x_right = (self.center['upper right'][0] + self.center['lower right'][0])/2
            y_upper = (self.center['upper left'][1] + self.center['upper right'][1])/2
            y_lower = (self.center['lower left'][1] + self.center['lower right'][1])/2

            is_center = np.zeros(len(self.data['tail 1']['x']))
            num_cross = 0
            self.num_cross = []  # number of times the animal crosses the border line of center area
            for idx in range(len(self.data['tail 1']['x'])):
                if self.data['tail 1']['x'][idx] > x_left and self.data['tail 1']['x'][idx] < x_right:
                    if self.data['tail 1']['y'][idx] > y_upper and self.data['tail 1']['y'][idx] < y_lower:
                        is_center[idx] = 1

                        if idx > 0:
                            if is_center[idx] != is_center[idx-1]:
                                num_cross+=1
                self.num_cross.append(num_cross)

            self.time_in_center = is_center
            self.cumu_time_center = []
            cumu = 0
            for f in range(self.nFrames):
                cumu += self.time_in_center[f]/self.fps
                self.cumu_time_center.append(cumu)

        else:
            print("please run moving_trace first")

    def plot_distance_to_center(self, t, savefigpath):
        # plot the distribution of distance to center
        # as well as a function of time
        distPlot = StartPlots()
        self.dist_center_bins = distPlot.ax.hist(self.dist_center, bins=np.linspace(0, 1200, 101))
        self.dist_center_bins_30 = distPlot.ax.hist(
            self.dist_center[0:30 * 60 * int(self.fps)],
            bins=np.linspace(0, 1200, 101)
        )
        distPlot.ax.set_xlabel('Distance from center (px)')
        distPlot.ax.set_ylabel('Occurance')
        distPlot.save_plot('Distribution of distance from center.tiff', 'tiff', savefigpath)

        # average distance from center in a running window
        window_frames = int(t * self.fps)
        n_running = self.nFrames - 1 - window_frames
        self.dist_center_running = np.zeros((n_running, 1))

        for ff in range(n_running):
            self.dist_center_running[ff] = np.nanmean(self.dist_center[ff:ff + window_frames])

        distRunningPlot = StartPlots()
        x = np.arange(len(self.dist_center_running)) / self.fps
        distRunningPlot.ax.plot(x, self.dist_center_running.flatten())
        distRunningPlot.ax.set_ylabel('Average distance from center (px)')
        distRunningPlot.ax.set_xlabel('Time (s)')

        distRunningPlot.save_plot('Average distance from center.tiff', 'tiff', savefigpath)
        plt.close('all')

    def read_data(self):
        data = {}
        if not hasattr(self, 't'):
            self.t = []

        if isinstance(self.filePath, str):
            with open(self.filePath) as csv_file:
                print("Loading data from: " + self.filePath)
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
                        self.nFrames += 1

                    else:
                        tempList = ['x', 'y', 'p']
                        for ii in range(len(row) - 1):
                            # get the corresponding body parts based on index
                            body = data['bodyparts'][int(np.floor((ii) / 3))]
                            data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                        #self.t.append(self.nFrames*(1/self.fps))
                        line_count += 1
                        self.nFrames += 1

                print(f'Processed {line_count} lines.')

                # add frame time
                #tStep= 1/self.fps
                data['time'] = self.t
                #self.t = np.array(self.t)
        else:
            data['time'] = np.nan
        # load rod speed data
        # check if rodPath is None
        if self.rodPath is not None:
            rodSpeed = pd.read_csv(self.rodPath, header=None)
            data['rodSpeed'] = rodSpeed.iloc[:, 0].values
            data['rodT'] = (rodSpeed.iloc[:, 1].values-self.t_start)/1000
        else:
            rodSpeed = None
            data['rodSpeed'] = rodSpeed
            data['rodT'] = None


        #%% for estimations with likelihood less than 0.8, replace the value with linear fit
        # based on previous and next value
        # corrected_data=copy.deepcopy(data)
        # kp_list = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot',
        #            'nose', 'left ear', 'right ear','left hand', 'right hand']
        # corrected_frames = []
        # for kp in kp_list:
        #     data[kp]['x'] = np.array(data[kp]['x'])
        #     data[kp]['y'] = np.array(data[kp]['y'])
        #     data[kp]['p'] = np.array(data[kp]['p'])
        #     for i in range(6, len(data['time'])-6):
        #         if data[kp]['p'][i] < 0.8:
        #             prev_reliable={}
        #             next_reliable = {}
        #             prev_reliable['x'] = data[kp]['x'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             prev_reliable['y'] = data[kp]['y'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             next_reliable['x'] = data[kp]['x'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             next_reliable['y'] = data[kp]['y'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             prev_reliable = pd.DataFrame(prev_reliable)
        #             next_reliable = pd.DataFrame(next_reliable)
        #             reliable_points = pd.concat([prev_reliable, next_reliable])
        #
        #             # If we found any reliable points, replace the unreliable x and y values
        #             if not reliable_points.empty:
        #                 if not i in corrected_frames:
        #                     corrected_frames.append(i)
        #                 # Calculate average x and y of these reliable points
        #                 avg_x = reliable_points['x'].mean()
        #                 avg_y = reliable_points['y'].mean()
        #
        #                 # Replace unreliable x and y values with the interpolated average
        #                 corrected_data[kp]['x'][i] = avg_x
        #                 corrected_data[kp]['y'][i] = avg_y
        #
        # # plot some frames to examine it
        # frame_num = 10198
        # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
        # plt.figure()
        # plt.imshow(curr_frame)
        # cmap = cm.get_cmap('viridis', len(kp_list))
        # for kp in kp_list:
        #     plt.scatter(data[kp]['x'][frame_num], data[kp]['y'][frame_num], c=cmap(kp_list.index(kp)), s=200,label = kp)
        #     plt.scatter(corrected_data[kp]['x'][frame_num],corrected_data[kp]['y'][frame_num],marker = 'x',c=cmap(kp_list.index(kp)), s=200,label = kp+'_corrected')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return data

    def check_quality(self):
        # go through the data and plot the distribution of p-values
        pass

    def get_movement(self):
        # calculate distance, running velocity, acceleration, based on tail (base of tail)
        savedatapath = os.path.join(self.analysis,'movement.pickle')
        if not os.path.exists(savedatapath):
            self.vel = np.zeros((self.nFrames-1, 1))
            self.dist = np.zeros((self.nFrames-1,1))
            self.accel = np.zeros((self.nFrames-1, 1))

            for ff in range(self.nFrames-1):
                self.dist[ff] = np.sqrt((self.data['tail 1']['x'][ff+1] - self.data['tail 1']['x'][ff])**2 +
                    (self.data['tail 1']['y'][ff + 1] - self.data['tail 1']['y'][ff]) ** 2)

                self.vel[ff] = (self.dist[ff])*self.fps
                if ff<self.nFrames-2:
                    self.accel[ff] = (self.vel[ff+1]-self.vel[ff])*self.fps
            # save vel, dist, accel in pickle file
            dist = self.dist
            vel = self.vel
            accel = self.accel
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist, vel, accel], f)
            f.close()
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist, self.vel, self.accel = pickle.load(f)
            f.close()

        if hasattr(self, 'px2cm'):
            self.dist = self.dist*self.px2cm
            self.vel = self.vel*self.px2cm
            self.accel = self.accel*self.px2cm

    def get_movement_running(self, t, savefigpath):
        savedatapath = os.path.join(self.analysis, 'movement_running.pickle')
        if not os.path.exists(savedatapath):
        # get average distance and velocity in running window of t seconds
            self.vel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.dist_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.accel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.dist_running[ff] = np.sum(self.dist[ff:ff+t*self.fps])
                self.vel_running[ff] = np.nanmean(self.vel[ff:ff+t*self.fps])
                self.accel_running[ff] = np.nanmean(self.accel[ff:ff+t*self.fps])
                self.vel[ff] = (self.dist[ff]) * self.fps

            dist_running = self.dist_running
            vel_running = self.vel_running
            accel_running = self.accel_running
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist_running, vel_running, accel_running], f)
            f.close()

            velPlot = StartPlots()
            x = np.arange(len(self.dist_running)) / self.fps
            velPlot.ax.plot(x, self.dist_running.flatten())
            velPlot.ax.set_ylabel('Average distance traveled (px)')
            #ax2 = velPlot.ax.twinx()
            #ax2.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.vel_running, color='red')
            #ax2.set_ylabel('Average velocity')
            velPlot.ax.set_xlabel('Time (s)')

            velPlot.save_plot('Running distance and velocity.png', 'png', savefigpath)
            plt.close(velPlot.fig)
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist_running, self.vel_running, self.accel_running = pickle.load(f)
            f.close()
        if hasattr(self, 'px2cm'):
            self.dist_running = self.dist_running*self.px2cm
            self.vel_running= self.vel_running*self.px2cm
            self.accel_running = self.accel_running*self.px2cm

    def get_angular_velocity(self):
        # calculate angular velocity based on tail and spine 1
        savedatapath = os.path.join(self.analysis, 'angular_velocity.pickle')
        if not os.path.exists(savedatapath):
            self.angVel = np.zeros((self.nFrames-1, 1))
            for ff in range(self.nFrames-1):
                y1 = self.data['spine 1']['y'][ff] - self.data['tail 1']['y'][ff]
                x1 = self.data['spine 1']['x'][ff] - self.data['tail 1']['x'][ff]

                y2 = self.data['spine 1']['y'][ff+1] - self.data['tail 1']['y'][ff+1]
                x2 = self.data['spine 1']['x'][ff+1] - self.data['tail 1']['x'][ff+1]

                self.angVel[ff] = self.get_angle([x1, y1], [x2, y2])*self.fps
            angVel = self.angVel
            with open(savedatapath, 'wb') as f:
                pickle.dump(angVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel = pickle.load(f)
            f.close()
        #self.angVel = self.angVel*self.fps

    def get_head_angular_velocity(self):
        savedatapath = os.path.join(self.analysis, 'head_angular_velocity.pickle')
        if not os.path.exists(savedatapath):
            self.headAngVel = np.zeros((self.nFrames, 1))
            for ff in range(self.nFrames-1):
                # get the mid point of two ears
                midX1 = (self.data['left ear']['x'][ff] + self.data['right ear']['x'][ff])/2
                midY1 = (self.data['left ear']['y'][ff] + self.data['right ear']['y'][ff])/2

                midX2 = (self.data['left ear']['x'][ff+1] + self.data['right ear']['x'][ff+1])/2
                midY2 = (self.data['left ear']['y'][ff+1] + self.data['right ear']['y'][ff+1])/2

                v1 = [self.data['nose']['x'][ff]-midX1, self.data['nose']['y'][ff]-midY1]
                v2 = [self.data['nose']['x'][ff+1]-midX2, self.data['nose']['y'][ff+1]-midY2]

                self.headAngVel[ff] = self.get_angle(v1, v2) * self.fps
            with open(savedatapath, 'wb') as f:
                pickle.dump(self.headAngVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.headAngVel = pickle.load(f)
            f.close()

    def get_stride(self,front_kp, back_kp, df_entry):
        savedatapath = os.path.join(self.analysis, 'stride_freq.csv')
        runFile = os.path.join(self.analysis, 'notExist.csv') # a not existing file to allow re-calculating
        if not os.path.exists(runFile):

            savefigpath = os.path.join(self.analysis)
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)

            # get rid the the turning period
            
            # %% define rod plane first
            # load reference point
            # calculate the average position
            tempData = self.data
            ave_left_rod_back = np.array([np.mean(np.array(tempData['rod_left_back']['x'])[np.array(tempData['rod_left_back']['p'])>0.95]),
                            np.mean(np.array(tempData['rod_left_back']['y'])[np.array(tempData['rod_left_back']['p'])>0.95])])
            ave_right_rod_back = np.array([np.mean(np.array(tempData['rod_right_back']['x'])[np.array(tempData['rod_right_back']['p'])>0.95]),
                            np.mean(np.array(tempData['rod_right_back']['y'])[np.array(tempData['rod_right_back']['p'])>0.95])])
            ave_center_rod_back = (ave_left_rod_back+ave_right_rod_back)/2
            self.data['ref_left_rod_back'] = ave_left_rod_back
            self.data['ref_right_rod_back'] = ave_right_rod_back
            self.data['ref_center_rod_back'] = ave_center_rod_back

            ave_left_rod_front = np.array([np.mean(np.array(tempData['rod_left_front']['x'])[np.array(tempData['rod_left_front']['p'])>0.95]),
                            np.mean(np.array(tempData['rod_left_front']['y'])[np.array(tempData['rod_left_front']['p'])>0.95])])
            ave_right_rod_front = np.array([np.mean(np.array(tempData['rod_right_front']['x'])[np.array(tempData['rod_right_front']['p'])>0.95]),
                            np.mean(np.array(tempData['rod_right_front']['y'])[np.array(tempData['rod_right_front']['p'])>0.95])])
            ave_center_rod_front = (ave_left_rod_front+ave_right_rod_front)/2

            self.data['ref_left_rod_front'] = ave_left_rod_front
            self.data['ref_right_rod_front'] = ave_right_rod_front
            self.data['ref_center_rod_front'] = ave_center_rod_front

            ave_left_rod_back = self.data['ref_left_rod_back']
            ave_right_rod_back = self.data['ref_right_rod_back']
            ave_center_rod_back = self.data['ref_center_rod_back']
            ave_left_rod_front = self.data['ref_left_rod_front']
            ave_right_rod_front = self.data['ref_right_rod_front']
            ave_center_rod_front = self.data['ref_center_rod_front']

            ref_plot = os.path.join(savefigpath, 'Rod coordinate.png')
            if not os.path.exists(ref_plot):
                frame = read_video(self.videoPath, 0, ifgray=False)
                # overlay the video frame?
                plt.figure()
                plt.imshow(frame)
                plt.scatter(self.data['rod_left_back']['x'], self.data['rod_left_back']['y'],
                            c=self.data['rod_left_back']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood
                plt.colorbar(label='Confidence')

                plt.scatter(self.data['rod_right_back']['x'], self.data['rod_right_back']['y'],
                            c=self.data['rod_right_back']['p'], cmap='viridis', s=100)

                plt.scatter(self.data['rod_left_front']['x'], self.data['rod_left_front']['y'],
                            c=self.data['rod_left_front']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood

                plt.scatter(self.data['rod_right_front']['x'], self.data['rod_right_front']['y'],
                            c=self.data['rod_right_front']['p'], cmap='viridis', s=100)

                # get average from keypoints with confidence higher than 95


                plt.scatter(ave_left_rod_back[0],ave_left_rod_back[1], s=500)
                plt.scatter(ave_right_rod_back[0], ave_right_rod_back[1], s=500)
                plt.scatter(ave_center_rod_back[0], ave_center_rod_back[1], s=500)

                plt.scatter(ave_left_rod_front[0],ave_left_rod_front[1], s=500)
                plt.scatter(ave_right_rod_front[0], ave_right_rod_front[1], s=500)
                plt.scatter(ave_center_rod_front[0], ave_center_rod_front[1], s=500)

                plt.savefig(os.path.join(savefigpath, 'rod_plane.png'))
                plt.close()
            # save the figure

            # %% examine the body parts in back view and front view

            # find behavior time (from rod start turning to fall)
            startTime= self.data['rodT'][self.data['rodSpeed_smoothed']>0][0]
            if np.isnan(self.data['rodStart'][0]):
                self.data['rodStart'][0] = 0
            endTime = startTime+df_entry['Performance'] + self.data['rodRun'][0] - self.data['rodStart'][0] # need the time stay on rod

            timeMaskDLC = np.logical_and(self.data['time']>=startTime, self.data['time']<= endTime)
            timeMaskRod = np.logical_and(self.data['rodT']>=startTime, self.data['rodT']<= endTime)
            nFramesInclude = np.sum(timeMaskDLC)
            time_include = self.data['time'][timeMaskDLC]
            kp_list = ['left hand', 'right hand', 'left foot', 'right foot']
            self.stride = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_front[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_front[1])**2)
                elif 'foot' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_back[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_back[1])**2)

            # try calculate the stride using distance from the rod (a horizontal line)
            self.stride_rod = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_left_rod_front, ave_right_rod_front)
                elif 'foot' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_right_rod_back, ave_left_rod_back)

            #tempMask = ~p_mask[timeMask]
            #    self.stride[tempMask,idx] = np.nan
            #    tempStride, tempIdx = fill_nans_and_split(self.stride[:, idx])
                # interpolate the nans
                # self.notnanChunks[kp] = tempIdx
                # for ich, chunk in enumerate(tempIdx):
                #     self.stride[chunk[0]:chunk[1]+1,idx] = tempStride[ich]


            # low-pass filter the data
            fps = 50
            self.t_interp = np.arange(time_include[0], time_include[-1] + 1 / fps, 1 / fps)
            self.filtered_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)
            self.interp_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)

            # need to determine the cutoff frequency here
            for idx, kp in enumerate(kp_list):
                #for ich, chunk in enumerate(self.notnanChunks[kp]):
                #    if chunk[1]-chunk[0]+1 > 18:  # padlen
                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
                self.interp_stride[:,idx] = np.interp(self.t_interp, time_include, self.stride_rod[:,idx])

                self.filtered_stride[:,idx] = butter_lowpass_filter(self.interp_stride[:,idx], 5,fps,order=5)

            #%%
            # examine the autocorrelation
            # average them over genotype and trial
            # find the time when rod speed reach 5/10
            #if df_entry['Trial']<=6:
            startSpeed = 5
            #else:
            #    startSpeed = 10
            startTime_auto = self.data['rodT'][self.data['rodSpeed_smoothed']>startSpeed][0]
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 subplots
            ax = ax.flatten()
            for ss in range(len(kp_list)):
                signal = pd.Series(self.filtered_stride[self.t_interp>startTime_auto,ss])
                autocorr_values = [signal.autocorr(lag=i) for i in range(len(signal)//2)]

                plot_time = 10
                # Subplot 1 (First row, spanning two columns)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, autocorr_values, linewidth=0.5)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, np.zeros(len(autocorr_values)),c='r', linewidth=2)
                #ax[ss].stem(range(len(autocorr_values)), autocorr_values,linefmt='b-', basefmt=" ", use_line_collection=True)
                ax[ss].set_title('Autocorrelation of ' + kp_list[ss])

                if ss==0:
                    # save autocorrelation value and lags in dataframe
                    autocorr_df = pd.DataFrame({'lags': np.arange(len(autocorr_values))/fps})
                autocorr_df[kp_list[ss]] = autocorr_values

            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis, 'Stride autocorrelation.png'), dpi=300)  # Save as PNG fil
            # save autocorrelation
            autocorr_df.to_csv(os.path.join(self.analysis, 'Stride autocorrelation.csv'))
            plt.close()

            #%%
            # instantaneous frequency with hilbert transform

            # Compute the analytic signal
            #
            # analytic_signal = hilbert(self.interp_stride[:,2])
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1 / fps))


            # Plot spectrogram

            # %% short time fourier transform
            # from scipy.signal import stft
            # frequencies, times, Zxx = stft(self.filtered_stride[:,3], fs=50, nperseg=256)
            # plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
            # plt.colorbar(label='Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [s]')
            # plt.title('STFT Magnitude')

            #%% pearson correlation between limbs
            # phase lag?
            # generate some plots
            pcorr = pd.DataFrame({'time': self.t_interp})
            corr_group = [['left hand','right hand'], ['left foot', 'right foot'],
                           ['left hand', 'left foot'], ['right hand', 'right foot']]
            corr_Idx = [[0,1], [2,3], [0,2], [1,3]]
            # xcorr between hands/feet/left/right
            timeStep = 2 # in second
            for kp_pairs,kp_idx in zip(corr_group,corr_Idx):
                corrCoeff_running = np.zeros((len(self.t_interp)))
                for idx,t in enumerate(self.t_interp):
                    tMask = np.logical_and(self.t_interp>t, self.t_interp <t+timeStep)
                    corrCoeff_running[idx] = np.corrcoef(self.filtered_stride[tMask,kp_idx[0]],
                                                             self.filtered_stride[tMask,kp_idx[1]])[0,1]
                pcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corrCoeff_running
            
            # cross correlation
            max_lag_sec = 1.0  # maximum lag to compute (in seconds)
            dt = self.t_interp[1] - self.t_interp[0]  # time step of your signal
            max_lag_samples = int(max_lag_sec / dt)

            # Store results
            max_xcorr = pd.DataFrame({'time': self.t_interp})
            max_lag = pd.DataFrame({'time': self.t_interp})

            for kp_pairs, kp_idx in zip(corr_group, corr_Idx):
                # Each element will be a 2D array: shape (len(t_interp), 2*max_lag_samples+1)
                # Arrays for max correlation and lag at each time point
                corr_max = np.full(len(self.t_interp), np.nan)
                lag_at_max = np.full(len(self.t_interp), np.nan)

                lags = np.arange(-max_lag_samples, max_lag_samples + 1) * dt

                for idx, t in enumerate(self.t_interp):
                    # 2-second window mask
                    tMask = (self.t_interp > t) & (self.t_interp < t + timeStep)
                    x = self.filtered_stride[tMask, kp_idx[0]]
                    y = self.filtered_stride[tMask, kp_idx[1]]

                    if len(x) < 2 or len(y) < 2:
                        continue

                    # Normalize signals
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Compute normalized cross-correlation
                    c = correlate(y, x, mode='full')
                    if np.std(x) != 0 and np.std(y) != 0:
                        c = c / (np.std(x) * np.std(y) * len(x))

                    # Center index
                    mid = len(c) // 2
                    c_window = c[mid - max_lag_samples: mid + max_lag_samples + 1]

                    # Find max correlation and corresponding lag
                    max_idx = np.nanargmax(c_window)
                    corr_max[idx] = c_window[max_idx]
                    lag_at_max[idx] = lags[max_idx]
                
                max_xcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corr_max
                max_lag[kp_pairs[0]+'-'+kp_pairs[1]] = lag_at_max

            #save cross correlation results
            pcorr_renamed = pcorr.copy()
            pcorr_renamed.columns = ['time'] + [col + '_pearson' for col in pcorr.columns[1:]]

            max_xcorr_renamed = max_xcorr.copy()
            max_xcorr_renamed.columns = ['time'] + [col + '_maxxcorr' for col in max_xcorr.columns[1:]]

            max_lag_renamed = max_lag.copy()
            max_lag_renamed.columns = ['time'] + [col + '_lag' for col in max_lag.columns[1:]]

            # 2. Merge all DataFrames on 'time'
            combined_df = pcorr_renamed.merge(max_xcorr_renamed, on='time').merge(max_lag_renamed, on='time')

            # 3. Save to CSV
            combined_df.to_csv(os.path.join(self.analysis, 'Stride correlation.csv'), index=False)

            # make a plot to show pearson correlation and cross correlation and max lag
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-right hand'], label= 'Hands')
            ax[1].plot(self.t_interp, pcorr['left foot-right foot'], label = 'Feet')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-right hand'], label= 'Hands')
            ax[2].plot(self.t_interp, max_xcorr['left foot-right foot'], label = 'Feet')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-right hand'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['left foot-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - HF.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            # same plot to show left and right
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-left foot'], label= 'Left')
            ax[1].plot(self.t_interp, pcorr['right hand-right foot'], label = 'Right')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-left foot'], label= 'LEft')
            ax[2].plot(self.t_interp, max_xcorr['right hand-right foot'], label = 'Right')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-left foot'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['right hand-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - LR.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            #%% calculate hand/foot step amplitude and frequency based on peak detection
            self.stride_amp = []
            self.stride_time = []
            self.stride_freq = np.full(self.filtered_stride.shape, np.nan)

            time = self.t_interp
            for ll in range(4): # step size and amplitude of 4 limbs
            # Detect peaks (foot lifts)
            
                distance = self.filtered_stride[:,ll]
                peaks, props = find_peaks(distance, prominence=2, distance=None)

                # Estimate baseline before each step using local minima
                inv_distance = -distance
                troughs, _ = find_peaks(inv_distance, prominence=2 / 2, distance=None)

                step_amplitudes = []
                step_times = []

                for peak in peaks:
                    # Find the closest following trough (baseline)
                    next_troughs = troughs[troughs > peak]
                    if len(next_troughs) == 0:
                        continue
                    baseline_idx = next_troughs[0]
                    amplitude = distance[peak] - distance[baseline_idx]
                    step_amplitudes.append(amplitude)
                    step_times.append(time[peak])

                step_amplitudes = np.array(step_amplitudes)
                step_times = np.array(step_times)

                self.stride_amp.append(step_amplitudes)
                self.stride_time.append(step_times)


                # Compute step frequency (Hz) in running 2 second window
                window = 2
                freqs = np.full(len(time), np.nan)  # preallocate

                for i, t in enumerate(time):
                    # count steps within [t - window/2, t + window/2]
                    mask = (step_times >= t - window/2) & (step_times <= t + window/2)
                    steps_in_window = step_times[mask]

                    if len(steps_in_window) >= 1:
                        intervals = np.diff(steps_in_window)
                        freqs[i] = 1 / np.mean(intervals)
                    else:
                        freqs[i] = np.nan

                self.stride_freq[:,ll] = freqs


            fig,ax = plt.subplots(5, 1, figsize=(16, 16))
            # rod speed
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 2, stride of hand
            ax[1].plot(self.t_interp, self.filtered_stride[:,0])
            ax[1].plot(self.t_interp , self.filtered_stride[:,1])
            ax[1].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Distance between left/right hand and the rod')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 3 foot
            ax[2].plot(self.t_interp, self.filtered_stride[:,2])
            ax[2].plot(self.t_interp, self.filtered_stride[:,3])
            ax[2].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].set_title('Distance between left/right foot and the rod')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 hand amplitude
            ax[3].stem(self.stride_time[0], self.stride_amp[0], linefmt='C0-',  basefmt=" ", label='left hand')
            ax[3].stem(self.stride_time[1], self.stride_amp[1],linefmt='C1-',  basefmt=" ",label='right hand')
            ax[3].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[3].set_title('Hand step amplitude')
            ax[3].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 (Third row, first column)
            ax[4].stem(self.stride_time[2], self.stride_amp[2], linefmt='C0-',  basefmt=" ", label='left foot')
            ax[4].stem(self.stride_time[3], self.stride_amp[3], linefmt='C1-',  basefmt=" ", label='right foot')
            ax[4].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[4].set_title('Foot step amplitude')

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
            
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis,'Stride amplitude.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            data = {'left hand': self.filtered_stride[:,0],
                    'right hand': self.filtered_stride[:,1],
                    'left foot': self.filtered_stride[:, 2],
                    'right foot': self.filtered_stride[:, 3],
                    'stride amplitude': self.stride_amp,
                    'stride time': self.stride_time,
                    'stride frequency': self.stride_freq,
                    'time': self.t_interp}
            #dataDF = pd.DataFrame(data)
            #dataDF.to_csv(savedatapath)
            # save to pickle file
            with open( os.path.join(self.analysis, 'stride_freq.pickle'), 'wb') as f:
                pickle.dump(data, f)

            #%%
            # cumulative area under the curve
            # cum_xcorr_foot = np.cumsum(xcorr['left foot-right foot'])/fps
            # cum_xcorr_hand = np.cumsum(xcorr['left hand-right hand']) / fps
            # cum_xcorr_left = np.cumsum(xcorr['left hand-left foot'])/fps
            # cum_xcorr_right = np.cumsum(xcorr['right hand-right foot']) / fps
            # plt.figure()
            # plt.plot(self.t_interp, cum_xcorr_foot)
            # plt.plot(self.t_interp, cum_xcorr_hand)
            # plt.plot(self.t_interp, cum_xcorr_left)
            # plt.plot(self.t_interp, cum_xcorr_right)
            # plt.plot(self.data['rodT'][timeMaskRod], self.data['rodSpeed_smoothed'][timeMaskRod])
            # plt.xlabel('time')
            # plt.ylabel('Cumulative area under the curve of xcorr')
            # plt.legend(['feet','hands','left','right', 'Rod speed'])
            # plt.savefig(os.path.join(self.analysis,'Stride correlation.png'), dpi=300)  # Save as PNG fil
            # #plt.show()
            # plt.close()
            # # cross correlation in 10 second window

            # # save data in csv
            # xcorr.to_csv(os.path.join(self.analysis, 'Stride crosscorrelation.csv'))

            #
            #%% tail angle
            # calculate spine 3 - tail 1 - tail 2 angle
            A = np.array([self.data['spine 3']['x'], self.data['spine 3']['y']]).T
            B = np.array([self.data['tail 1']['x'], self.data['tail 1']['y']]).T
            C = np.array([self.data['tail 2']['x'], self.data['tail 2']['y']]).T

            A = A[timeMaskDLC,:]
            B = B[timeMaskDLC,:]
            C = C[timeMaskDLC,:]
            # Calculate vectors AB and BC
            AB = B - A
            BC = C - B

            # Calculate the angle between AB and BC
            # Calculate dot and cross products for each time point
            dot_product = np.sum(AB * BC, axis=1)  # Dot product for each row (time point)
            cross_product = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]  # Cross product for each time point

            # Calculate the angle at each time point
            angles = np.arctan2(cross_product, dot_product)

            # Convert to degrees
            angles = np.degrees(angles)

            # interpolate and filter the angle
            fps = 50
            self.filtered_angle = np.full((len(self.t_interp)), np.nan)
            self.interp_angle = np.full((len(self.t_interp)), np.nan)

            # need to determine the cutoff frequency here

                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
            self.interp_angle= np.interp(self.t_interp, time_include, angles)

            self.filtered_angle = butter_lowpass_filter(self.interp_angle, 5,fps,order=5)


            # save data in csv
            tail_angle= pd.DataFrame({'angle':self.filtered_angle, 'time':self.t_interp})
            tail_angle.to_csv(os.path.join(self.analysis, 'Tail angle.csv'))
            # Calculate the angle in radians using atan2 for correct sign

            # plot the video frame with keypoint estimatino
            # frame_num = 7760
            # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
            # plt.figure()
            # plt.imshow(curr_frame)
            # kp_plot = ['tail 2']
            # for kp in kp_plot:
            #     plt.scatter(self.data[kp]['x'][frame_num], self.data[kp]['y'][frame_num], s=20)

            #%% head angle

            #%% tail position
            # plot the density distribution of the tail
            # set coordinate of tail 1 to be (0, 0)
            # ego_tail = {}
            # tail_key = ['tail 1', 'tail 2', 'tail 3']
            # for t in tail_key:
            #     ego_tail[t] = {}
            #     ego_tail[t]['x']= np.array(self.data[t]['x'])-np.array(self.data['tail 1']['x'])
            #     ego_tail[t]['y'] = np.array(self.data[t]['y']) - np.array(self.data['tail 1']['y'])
            #
            # plt.figure(figsize=(12, 6))
            # # Density plot for aligned b coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 2']), x='x', y='y',
            #             fill=True, cmap='Blues', alpha=0.5, label='Point B',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm())
            # # Density plot for aligned c coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 3']), x='x', y='y',
            #             fill=True, cmap='Reds', alpha=0.5, label='Point C',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm()
            #             )
            # plt.axhline(0, color='black', lw=1, ls='--', label='y = 0')
            # plt.axvline(0, color='black', lw=1, ls='--', label='x = 0')
            #
            # plt.title('Density Distribution of Aligned Points B and C')
            # plt.xlabel('Aligned B X')
            # plt.ylabel('Aligned B Y / Aligned C Y')
            # plt.axhline(0, color='black', lw=0.5, ls='--')
            # plt.axvline(0, color='black', lw=0.5, ls='--')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # with open(savedatapath, 'wb') as f:
            #     pickle.dump(self.stride, self. f)
            # f.close()
        else:
            print("Analysis already done")
            return np.nan

    def get_angular_velocity_running(self, t, savefigpath):
        # calculate angular velocity with running window t
        savedatapath = os.path.join(self.analysis, 'angular_velocity_running.pickle')

        if not os.path.exists(savedatapath):
            self.angVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.headAngVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.angVel_running[ff] = np.nanmean(self.angVel[ff:ff+t*self.fps])
                self.headAngVel_running[ff] = np.nanmean(self.headAngVel[ff:ff+t*self.fps])

            # plot the velocity here
            angPlot = StartPlots()
            x = np.arange(len(self.angVel_running)) / self.fps
            angPlot.ax.plot(x, self.angVel_running.flatten())
            angPlot.ax.set_ylabel('Angular velocity')
            ax2 = angPlot.ax.twinx()
            ax2.plot(x, self.headAngVel_running.flatten(), color='red')
            ax2.set_ylabel('Head angular velocity', color='red')
            angPlot.ax.set_xlabel('Time (s)')

            angPlot.save_plot('Running angular vel.tiff', 'tiff', savefigpath)
            plt.close(angPlot.fig)

            angVel_running = self.angVel_running
            headAngVel_running = self.headAngVel_running
            with open (savedatapath, 'wb') as f:
                pickle.dump([angVel_running,headAngVel_running], f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel_running, self.headAngVel_running = pickle.load(f)
            f.close()

    def get_angle(self, v1, v2):
        # get angle between two vectors
            v1_u = self.unit_vector(v1)
            v2_u = self.unit_vector(v2)

            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            if v1[0] * v2[1] - v1[1] * v2[0] < 0:
                angle = -angle
            return angle

    def plot_keypoints(self, nFrame):
        bodyparts = self.data['bodyparts']
        skeleton = [
            ['nose', 'head'],
            ['head', 'left ear'],
            ['head', 'right ear'],
            ['head', 'spine 1'],
            ['spine 1', 'left hand'],
            ['spine 1', 'right hand'],
            ['spine 1', 'spine 2'],
            ['spine 2', 'spine 3'],
            ['spine 3', 'left foot'],
            ['spine 3', 'right foot'],
            ['spine 3', 'tail 1'],
            ['tail 1', 'tail 2'],
            ['tail 2', 'tail 3']
        ]
        image = read_video(self.videoPath, nFrame, ifgray=True)
        plt.imshow(image)
        for bd in bodyparts:
            plt.scatter(self.data[bd]['x'][nFrame], self.data[bd]['y'][nFrame])
        for sk in skeleton:
            plt.plot([self.data[sk[0]]['x'][nFrame],self.data[sk[1]['x'][nFrame]]], [self.data[sk[0]]['y'][nFrame],self.data[sk[1]['y'][nFrame]]])

        plt.show()

    def unit_vector(self, v):
        """ Returns the unit vector of the vector.  """
        return v / np.linalg.norm(v)

def read_video(videoPath, frame, ifgray):
    # ifgray: if convert the image to grayscale
    vid = imageio.get_reader(videoPath)

        #for ii in tqdm(range(self.nFrames)):
    if ifgray:
        image = color.rgb2gray(vid.get_data(frame))
    else:
        image = vid.get_data(frame)
        #   [xdim, ydim] = image.shape
        #    if ii == 0:
        #        # get video dimensions
                #imageStack = np.zeros((xdim, ydim, self.nFrames))
        #        imageStack = []
        #    imageStack.append(image)
    return image

def frame_input(videoPath):

    frame = read_video(videoPath, 0, ifgray=True)
    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.axis('off')

    ax.set_title('Please select 4 cornors, upper L -> upper R -> lower R -> lower L')
    points = []
    point_names = ['upper left', 'upper right', 'lower right', 'lower left']

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')

            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Add a button to confirm input after 4 clicks
    button_ax = plt.axes([0.8, 0.05, 0.1, 0.02])  # Button position [x, y, width, height]
    button = plt.Button(button_ax, 'Confirm')

    confirm_clicked = False

    def confirm_callback(event):
        global confirm_clicked
        if event.inaxes == button_ax:
            confirm_clicked = True
            print(confirm_clicked)
            plt.disconnect(cid)  # Disconnect the onclick event handler function
            plt.close()  # Close the plot window

    button.on_clicked(confirm_callback)

    plt.show(block=True)

    arena = {}
    for i, n in enumerate(point_names):
        arena[n] = points[i]

    return arena

def map_point(F1, F2, F3, B1, B2):
    """ utils function used to calculate the points in one view 
    based on reference points in both views, and the point in the other view"""
    F1 = np.array(F1)
    F2 = np.array(F2)
    F3 = np.array(F3)

    B1 = np.array(B1)
    B2 = np.array(B2)

    vF = F2 - F1
    vB = B2 - B1

    # coordinates in front view
    u = np.dot(F3-F1, vF) / np.dot(vF, vF)

    cross = np.cross(vF, F3-F1)
    w = cross / np.dot(vF, vF)

    # perpendicular direction in back view
    perpB = np.array([-vB[1], vB[0]])

    B3 = B1 + u*vB + w*perpB

    return B3

def correct_bodyparts(df, 
                      ref_bp = ['rod_left_back', 'rod_right_back', 'rod_left_front', 'rod_right_front'],
                      image_width = 1596):
    """ Corrects body part positions in the DataFrame. 
    if a bodypart jumped too far away to another half of the frame, move it back
    based on the reference point
    for rotarod data only
    """
    # input:
    # df: DataFrame containing body part positions
    # ref_bp: list of reference body part names
    # image_width: width of the image in pixels

    ref = {}
    df_corrected = df.copy()
    for bp in ref_bp:
        ref[bp] = {}
        x_col = df.columns[(df.iloc[0] == bp) & (df.iloc[1] == 'x')][0]
        y_col= df.columns[(df.iloc[0] == bp) & (df.iloc[1] == 'y')][0]
        x = df.loc[2:, x_col].astype(float).to_numpy()
        y = df.loc[2:, y_col].astype(float).to_numpy()

        # estimate the position by taking the 5%-95% percentiles, and taking average
        ref[bp]['x'] = np.mean(x[np.logical_and(x >= np.percentile(x, 10), x <= np.percentile(x, 90))])
        ref[bp]['y'] = np.mean(y[np.logical_and(y >= np.percentile(y, 10), y <= np.percentile(y, 90))])

        #%% for test
        videofilePath = r'Y:\HongliWang\Rotarod\ASD_strains\TSC2_adol\DLCforMoseq\ASD578_251217_trial12025-12-17T14_41_40DLC_resnet50_rotarodNov5shuffle1_300000_filtered_clip3.mp4'
        image = read_video(videofilePath, 0, ifgray=True)
    
    # for all other bodyparts, look for outliers that is in another half of the frame
    bodyparts = np.unique(df.iloc[0,1:].tolist())
    for bp in bodyparts:
        if bp not in ref_bp:
            x_col = df.columns[(df.iloc[0] == bp) & (df.iloc[1] == 'x')][0]
            y_col = df.columns[(df.iloc[0] == bp) & (df.iloc[1] == 'y')][0]
            x = df.loc[2:, x_col].astype(float).to_numpy()
            y = df.loc[2:, y_col].astype(float).to_numpy()
            
            # determine if x in left or right half of the frame
            x_mean = np.mean(x[np.logical_and(x >= np.percentile(x, 10), x <= np.percentile(x, 90))])

            if x_mean < image_width / 2:
                # x is in the left half of the frame
                # look for outliers in the right half of the frame
                outlier_mask = x > image_width / 2
                ref1 = [ref['rod_left_front']['x'], ref['rod_left_front']['y']]
                ref2 = [ref['rod_right_front']['x'], ref['rod_right_front']['y']]
                ref3 = [ref['rod_left_back']['x'], ref['rod_left_back']['y']]
                ref4 = [ref['rod_right_back']['x'], ref['rod_right_back']['y']]
            else:
                # x is in the right half of the frame
                outlier_mask = x < image_width / 2
                ref3 = [ref['rod_left_front']['x'], ref['rod_left_front']['y']]
                ref4 = [ref['rod_right_front']['x'], ref['rod_right_front']['y']]
                ref1 = [ref['rod_left_back']['x'], ref['rod_left_back']['y']]
                ref2 = [ref['rod_right_back']['x'], ref['rod_right_back']['y']]


            # for outliers, calculate the corrected position given the reference point that
            # rod_left_back = rod_left_front, and rod_right_back = rod_right_front
            outlier_index = np.where(outlier_mask)[0]
            
            for oidx in outlier_index:
                target_point = [x[oidx], y[oidx]]
                corrected_point = map_point(ref1, ref2, target_point, ref3, ref4)
                df_corrected.loc[2 + oidx, x_col] = str(corrected_point[0])
                df_corrected.loc[2 + oidx, y_col] = str(corrected_point[1])

    return df_corrected