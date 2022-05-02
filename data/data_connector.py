

class DatasetConnector:

    def __init__(self):
        self.rss = None
        self.pos = None
        self.floor = None
        self.floors = None
        self.num_floors = None

        self.floorplan_width = None
        self.floorplan_height = None

        self.split_indices = None

    def load_dataset(self):
        pass

    def get_dataset_identifier(self):
        raise NotImplementedError
