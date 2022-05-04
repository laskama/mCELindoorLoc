import numpy as np

from data.base_data_provider import BaseDataProvider
from visualization.floor_plan_plot import FloorplanPlot
from utils.definitions import get_project_root
import seaborn as sns

root = get_project_root()


def visualize_output(polys, floors, model_names, dp: BaseDataProvider, shuffle=False, seq_ids=None):
    dataset = dp.dc.get_dataset_identifier()
    if dataset == 'gia_vslam':
        img_base = root + "/datasets/giaIndoorLoc/floor_{}/floorplan.jpg"
    elif dataset == 'tampere':
        img_base = root + "/datasets/tampere/floorplan/{}OG.png"
    else:
        raise NotImplementedError('You have specified the --visualize_prediction flag, '
                                  'but there is no visualization option for dataset: {}'.format(dataset))

    true_pos = dp.get_data(dp.pos, 'test')

    colors = sns.color_palette("deep")

    if seq_ids is None:
        seq_ids = np.arange(len(floors))

    if shuffle:
        np.random.shuffle(seq_ids)

    for plot_idx, s_id in enumerate(seq_ids):

        floor = floors[s_id]
        f = int(floor)

        fp_dims = (dp.floorplan_width[f], dp.floorplan_height[f])
        fp = FloorplanPlot(fp_dims, floorplan_bg_img=img_base.format(dp.floors[f]))

        # draw ground truth
        fp.draw_points(true_pos[s_id, 0], true_pos[s_id, 1], color='green', label='Ground Truth Position')

        # draw predictions
        for idx, mp in enumerate(polys):
            poly = mp[s_id]
            fp.draw_points(poly[0], poly[1], color=colors[idx], label=model_names[idx])

        fp.set_title('Floor: {}'.format(dp.floors[f]))
        fp.axis.axis('off')
        fp.axis.legend()

        fp.show_plot()
