#%% analyze longitudinal registered calcium imaging data

class Longitudinal:
    def __init__(self, root_dir, Odor, rotarod):

        # take odor and rotarod imaging object as input
        self.root_dir = root_dir
        self.Odor = Odor
        self.rotarod = rotarod
    
    def load_data(self):
        # load the longitudinal registered data, including the registered images, the extracted df/f traces, and the aligned behavior events
        # go through each animal, find the longitudinal registered data, and for each session, find the referecence field of view
        # reference FOV: std projection and correlation FOV

        # concatenate data_index from odor and rotarod together, add one column to indicate the task
        x=1