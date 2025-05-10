import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

class ScrollableScanViewer:
    def __init__(self, volume, title, axis=2):
        self.volume = volume
        self.axis = axis
        self.slice = volume.shape[axis] // 2
        self.n_slices = volume.shape[axis]

        self._updating_slider = False 

        # Create figure and adjust for slider
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.15)

        # Show initial slice
        self.im = self.display_slice()
        self.ax.set_title(f'{title}\nSlice {self.slice} (axis={self.axis})')
        self.ax.axis('off')

        # Create slider
        self.ax_slider = self.fig.add_axes([0.25, 0.05, 0.5, 0.03])
        self.slider = Slider(self.ax_slider, 'Slice', 0, self.n_slices - 1,
                             valinit=self.slice, valfmt='%0.0f')
        self.slider.on_changed(self.update_from_slider)

        # Connect scroll event
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def get_slice(self, index):
        if self.axis == 0:
            return self.volume[index, :, :]
        elif self.axis == 1:
            return self.volume[:, index, :]
        else:
            return self.volume[:, :, index]

    def display_slice(self):
        return self.ax.imshow(self.get_slice(self.slice), cmap='gray', origin='lower')

    def update_slice(self, new_index, from_slider=False):
        self.slice = int(np.clip(new_index, 0, self.n_slices - 1))
        self.im.set_data(self.get_slice(self.slice))
        self.ax.set_title(f'Slice {self.slice} (axis={self.axis})')
        self.fig.canvas.draw_idle()

        if not from_slider:
            self._updating_slider = True
            self.slider.set_val(self.slice)
            self._updating_slider = False

    def onscroll(self, event):
        if event.button == 'up':
            self.update_slice(self.slice + 1)
        elif event.button == 'down':
            self.update_slice(self.slice - 1)

    def update_from_slider(self, val):
        if not self._updating_slider:
            self.update_slice(int(val), from_slider=True)