import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def ScrollableScanViewer(volume, title, match_coords=None, axis=2):
    '''
    View the 3D MRI scan with a slider.

    Optionally, can pass match_coords which is a list of detected tumours.
    '''
    match_coords = match_coords if match_coords is not None else []
    max_index = volume.shape[axis] - 1

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_title(title)
    ax.axis('off')

    index = max_index // 2

    # Helper to extract slice and overlay points
    def get_slice_and_overlay(idx):
        if axis == 0:
            img = volume[idx, :, :]
            pts = [(y, x) for x, y, z in match_coords if z == idx]
        elif axis == 1:
            img = volume[:, idx, :]
            pts = [(z, x) for x, y, z in match_coords if y == idx]
        else:
            img = volume[:, :, idx]
            pts = [(x, y) for x, y, z in match_coords if z == idx]
        return img.T, pts

    # Initial view
    img, pts = get_slice_and_overlay(index)
    im_display = ax.imshow(img, cmap='gray', origin='lower')
    scatter = ax.plot(*zip(*pts), 'ro')[0] if pts else ax.plot([], [], 'ro')[0]
    coord_text = ax.text(0.02, 0.98, f"Slice: {index}/{max_index}", color='white',
                         transform=ax.transAxes, ha='left', va='top', fontsize=10,
                         bbox=dict(facecolor='black', alpha=0.5))

    # Slider widget
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, f"Slice (axis {axis})", 0, max_index, valinit=index, valstep=1)

    def update_display(idx):
        idx = int(np.clip(idx, 0, max_index))
        img, pts = get_slice_and_overlay(idx)
        im_display.set_data(img)
        if pts:
            xs, ys = zip(*pts)
            scatter.set_data(xs, ys)
        else:
            scatter.set_data([], [])
        coord_text.set_text(f"Slice: {idx}/{max_index}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        step = 1 if event.button == 'up' else -1
        new_idx = int(np.clip(slider.val + step, 0, max_index))
        slider.set_val(new_idx)  # this triggers the update once

    def on_key(event):
        if event.key in ['right', 'left']:
            step = 1 if event.key == 'right' else -1
            new_idx = int(np.clip(slider.val + step, 0, max_index))
            slider.set_val(new_idx)

    slider.on_changed(update_display)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()