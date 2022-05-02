import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.image import AxesImage


class FloorplanPlot:
    def __init__(self, floorplan_dimensions, floorplan_bg_img="",
                 xtick_freq=None, background_alpha=0.3, sub_plots=None, prreset_fig=None):

        self.floorplan_dimensions = floorplan_dimensions
        self.floorplan_bg_image = floorplan_bg_img

        self.axis: Axes = None
        self.fig: Figure = None

        self.bg_img: AxesImage = None
        self.bg_alpha = background_alpha
        self.sub_plots = sub_plots
        self.preset_fig = prreset_fig

        self.draw_background()

        if xtick_freq is not None:
            self.set_tick_frequency(xtick_freq)

    def show_plot(self):
        plt.show()

    def init_plot(self):
        if self.sub_plots is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = self.preset_fig
            ax = plt.subplot(*self.sub_plots)

        self.axis = ax
        self.fig = fig

    def set_title(self, title="title"):
        self.axis.set_title(title)

    def draw_background(self):
        if self.axis is None:
            self.init_plot()
        # bg image computations
        try:
            bg_image = plt.imread(self.floorplan_bg_image)
            self.bg_img = self.axis.imshow(bg_image, extent=[0, 0 + self.floorplan_dimensions[0],
                                                             0, 0 + self.floorplan_dimensions[1]],
                                           alpha=self.bg_alpha)
        except FileNotFoundError:
            print("No background image found")

    def draw_points(self, x_points, y_points, color='b', alpha=1, **kwargs):
        if self.axis is None:
            self.init_plot()
        # plot raw points
        self.axis.scatter(x_points, y_points, color=color, alpha=alpha, **kwargs)

    def set_tick_frequency(self, frequency=1.0):
        self.axis.set_xticks(np.arange(0, self.floorplan_dimensions[0], frequency))
        self.axis.set_yticks(np.arange(0, self.floorplan_dimensions[1], frequency))
