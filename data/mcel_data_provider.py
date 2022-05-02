import numpy as np
from data.base_data_provider import BaseDataProvider


class MCELdataProvider(BaseDataProvider):

    def __init__(self, params, dc):

        super().__init__(params, dc)

        self.aug_encoding = None
        self.grid_per_floor = None

        self.multi_labels = None
        self.multi_grid_cell_labels = None
        self.multi_labels_no_pad = None
        self.multi_grid_cell_labels_no_pad = None

        # have to be stored for evaluation
        self.grid_labels = None

    def get_num_grid_cells(self):
        return np.sum(np.array(self.grid_per_floor))

    def transform_to_grid_encoding(self):

        grid_labels = np.zeros((len(self.pos), 3))
        g_cnt = 0
        grid_per_floor = []
        aug_encoding = {}

        # transform per floor since each floor might have different dimensions
        for f_idx in range(self.num_floors):

            # get subset of labels for given floor
            sub_idx = np.where(self.floor == self.floors[f_idx])[0]
            labels = self.pos[sub_idx]

            enc, enc_aug, num = self.transform_to_grid_encoding_(
                labels, self.floorplan_width[f_idx], self.floorplan_height[f_idx], grid_offset=g_cnt)

            # store how many grid cells where used for the floor (required for reverse transformation)
            grid_per_floor += [num]

            # set global grid idx (unique across all floors)
            enc[:, 2] += g_cnt
            g_cnt += num

            # convert local aug encoding of floor to global idx
            for k, v in enc_aug.items():
                glob_idx = sub_idx[k]
                aug_encoding[glob_idx] = v

            grid_labels[sub_idx, :] = enc

        self.aug_encoding = aug_encoding
        self.grid_per_floor = grid_per_floor
        self.y = grid_labels
        self.grid_labels = self.y

        return self

    def transform_to_grid_encoding_(self, labels, width, height, grid_offset):

        grid_size = self.pr.get_param('grid_size')
        padding_ratio = self.pr.get_param('padding_ratio')

        origins, num = compute_grid_cell_origins_of_encoding(width, height, grid_size)

        # compute the distance of each position to the origins of the grid cells
        dist_to_center = np.full((len(labels), len(origins)), np.inf)
        dist_to_center_cmp = np.full((len(labels), len(origins), 2), np.inf)
        for o_idx, o in enumerate(origins):
            d = np.linalg.norm(labels - o, axis=1)
            dist_to_center[:, o_idx] = d
            dist_to_center_cmp[:, o_idx, :] = np.abs(labels - o)

        # obtain the grid cells that are closest to the positions
        resp_grid_idx = np.argmin(dist_to_center, axis=1)
        resp_origins = origins[resp_grid_idx, :]

        # find alternative origins for augmentation
        alt_origin = np.array(
            np.where(np.all(dist_to_center_cmp[:, :, :] < (grid_size / 2 + grid_size * padding_ratio), axis=2)))

        # use alternative grid cells (within padding zone) to construct additional augmented label encoding
        aug_encoding = {}
        for l_idx in range(len(labels)):
            mask = np.where(alt_origin[0, :] == l_idx)[0]
            aug_origins = alt_origin[1, mask]
            aug_idx = aug_origins[np.where(aug_origins != resp_grid_idx[l_idx])[0]]
            if len(aug_idx) > 0:
                aug_enc = np.zeros((len(aug_idx), 3))
                aug_enc[:, 2] = aug_idx + grid_offset
                aug_enc[:, :2] = (labels[l_idx] - origins[aug_idx, :]) / (
                        grid_size / 2.0 + grid_size * padding_ratio)

                aug_encoding[l_idx] = aug_enc

        # determine final encoding
        grid_encoding = np.zeros((len(labels), 3))
        grid_encoding[:, 2] = resp_grid_idx
        grid_encoding[:, :2] = (labels - resp_origins) / (grid_size / 2.0 + grid_size * padding_ratio)

        return grid_encoding, aug_encoding, num

    def compute_multilabel_aug_data(self, weighted_grid_labels=True):
        # get data and labels according to generator mode and k-fold split idx
        labels = self.y

        # setup multi-labels
        num_grids = int(self.get_num_grid_cells())
        multi_labels = np.zeros((len(labels), num_grids, 3))
        multi_grid_cell_labels = np.zeros((len(labels), num_grids))
        multi_grid_cell_labels_no_pad = np.zeros((len(labels), num_grids))
        multi_labels_no_pad = np.zeros((len(labels), num_grids, 3))

        for idx in range(len(labels)):

            indices = [int(labels[idx, 2])]
            c_dist = np.sum(np.square(labels[idx, :2]))

            multi_labels[idx, int(labels[idx, 2]), :2] = labels[idx, :2]
            multi_labels[idx, int(labels[idx, 2]), 2] = c_dist
            multi_grid_cell_labels[idx, int(labels[idx, 2])] = c_dist
            multi_grid_cell_labels_no_pad[idx, int(labels[idx, 2])] = 1.0

            multi_labels_no_pad[idx, int(labels[idx, 2]), :2] = labels[idx, :2]
            multi_labels_no_pad[idx, int(labels[idx, 2]), 2] = 1.0

            if idx in self.aug_encoding:
                for aug in self.aug_encoding[idx]:
                    multi_labels[idx, int(aug[2]), :2] = aug[:2]
                    c_dist = np.sum(np.square(aug[:2]))

                    multi_labels[idx, int(aug[2]), 2] = c_dist
                    multi_grid_cell_labels[idx, int(aug[2])] = c_dist
                    indices.append(int(aug[2]))
            else:
                multi_labels[idx, int(labels[idx, 2]), 2] = 1.0
                multi_grid_cell_labels[idx, int(labels[idx, 2])] = 1.0

            # inverse &normalize and inverse
            multi_labels[idx, np.array(indices), 2] = 1.0 / multi_labels[idx, np.array(indices), 2]
            d_sum = np.sum(multi_labels[idx, np.array(indices), 2])
            multi_labels[idx, np.array(indices), 2] /= d_sum

            if not weighted_grid_labels:
                multi_labels[idx, np.array(indices), 2] = 1

            multi_grid_cell_labels[idx, np.array(indices)] = 1.0 / multi_grid_cell_labels[
                idx, np.array(indices)]
            d_sum = np.sum(multi_grid_cell_labels[idx, np.array(indices)])
            multi_grid_cell_labels[idx, np.array(indices)] /= d_sum

        # reshape labels
        multi_labels = np.reshape(multi_labels,
                                  [len(multi_labels), num_grids * 3])
        multi_labels_no_pad = np.reshape(multi_labels_no_pad,
                                         [len(multi_labels_no_pad),
                                          num_grids * 3])

        self.multi_labels = multi_labels
        self.multi_grid_cell_labels = multi_grid_cell_labels
        self.multi_labels_no_pad = multi_labels_no_pad
        self.multi_grid_cell_labels_no_pad = multi_grid_cell_labels_no_pad

        return self

    def get_output_dim(self):
        num_grid_cells = np.shape(self.y[0])[1]
        return [num_grid_cells, num_grid_cells * 2]

    def get_y(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]

        return [self.y[idx][subset] for idx in range(2)]

    def set_labels(self, class_pad=True, reg_pad=True):
        if class_pad:
            y_c = self.multi_grid_cell_labels
        else:
            y_c = self.multi_grid_cell_labels_no_pad

        if reg_pad:
            y_r = self.multi_labels
        else:
            y_r = self.multi_labels_no_pad

        self.y = [y_c, y_r]

        return self

    def decode_grid_labels(self, grid_pred, within_grid_reg, true_grid_enc, true_pos):
        decoded_preds = np.zeros((len(grid_pred), 5))
        decoded_true = np.zeros((len(grid_pred), 5))

        # get floor predictions
        chosen_cells = np.argmax(grid_pred, axis=1)
        decoded_preds[:, 4] = chosen_cells
        decoded_true[:, 4] = true_grid_enc[:, 2]

        floor_ids = self.get_floors_of_grid_cells(chosen_cells)
        floor_ids_true = self.get_floors_of_grid_cells(true_grid_enc[:, 2])
        offset = 0

        for f_idx in range(self.num_floors):
            sub_idx_pred = np.where(floor_ids == f_idx)[0]
            sub_idx_true = np.where(floor_ids_true == f_idx)[0]

            # offset required for obtaining local grid cell idx of given floor
            if f_idx > 0:
                offset = np.cumsum(self.grid_per_floor)[f_idx - 1]

            # decode the prediction into global coordinate system
            decoded_labels = self.convert_from_2dim_overlapping_grid(
                grid_pred[sub_idx_pred], within_grid_reg[sub_idx_pred], offset=offset,
                height=self.floorplan_height[f_idx], width=self.floorplan_width[f_idx])

            decoded_preds[sub_idx_pred, :2] = decoded_labels
            decoded_preds[sub_idx_pred, 2] = 0  # only single building supported here
            decoded_preds[sub_idx_pred, 3] = f_idx

            decoded_true[sub_idx_true, :2] = true_pos[sub_idx_true, :2]
            decoded_true[sub_idx_true, 2] = 0   # only single building supported here
            decoded_true[sub_idx_true, 3] = f_idx

        return decoded_preds, decoded_true

    def convert_from_2dim_overlapping_grid(self, grid_pred, within_cell_reg, width, height, offset=0):
        grid_size = self.pr.get_param('grid_size')
        padding_ratio = self.pr.get_param('padding_ratio')

        # obtain the origins of the grid cells for decoding
        origins, _ = compute_grid_cell_origins_of_encoding(width, height, grid_size)

        # determine the grid cell idx with the highest probability
        chosen = np.argmax(grid_pred, axis=1)

        # determine local grid_idx from global one
        gs_choice = chosen - offset

        # will hold the decoded predictions
        pred_fold = np.zeros((len(grid_pred), 2))

        for idx in range(len(grid_pred)):
            # get within grid cell regression
            encoded_pred = within_cell_reg[idx, chosen[idx] * 2:(chosen[idx] + 1) * 2]

            # decode to global coordinates
            decoded_pred = origins[gs_choice[idx], :] + encoded_pred[:2] * (grid_size / 2.0 + grid_size * padding_ratio)

            pred_fold[idx, :] = decoded_pred

        return pred_fold

    def get_floors_of_grid_cells(self, g_ids):
        # computes the floor from the global grid cell idx
        # using the amount of grid cells per floor
        agg = np.cumsum(self.grid_per_floor)
        test = (g_ids.reshape(-1, 1) < agg.reshape(1, -1)).astype(float)
        row_sum = np.sum(test, axis=1)

        f_idx = (len(agg) - row_sum)

        return f_idx

    def calc_performance(self, y_pred_grid, y_pred_box, y_true_grid_enc, y_true_pos, compute_error_vec=False):
        metrics = {}

        y_pred, y_true_grid_enc = self.decode_grid_labels(y_pred_grid, y_pred_box, y_true_grid_enc, y_true_pos)

        bld_pred = y_pred[:, 2]
        bld_true = y_true_grid_enc[:, 2]
        # bld_acc = np.where(bld_pred == bld_true)[0]

        floor_pred = y_pred[:, 3]
        floor_true = y_true_grid_enc[:, 3]
        floor_acc = np.where(floor_pred == floor_true)[0]

        grid_cell_acc = np.where(y_pred[:, 4] == y_true_grid_enc[:, 4])[0]

        error_vec = np.linalg.norm(y_pred[:, :2] - y_true_grid_enc[:, :2], axis=1)

        pos_error = np.mean(error_vec)
        pos_error_median = np.median(error_vec)

        mask_floor = np.logical_and(bld_pred == bld_true,
                                    floor_pred == floor_true)

        pos_error_correct_floor, pos_error_wrong_floor, _, _ = compute_acc_of_hit_and_no_hit(
            mask_floor, y_pred, y_true_grid_enc)

        # correct_idx_f = np.where(mask_floor)[0]
        # wrong_idx_f = np.where(~mask_floor)[0]
        #
        # pos_error_correct_floor = np.mean(
        #     np.linalg.norm(
        #         y_pred[correct_idx_f, :2] - y_true_grid_enc[correct_idx_f, :2],
        #         axis=1))
        #
        # pos_error_wrong_floor = np.mean(
        #     np.linalg.norm(y_pred[wrong_idx_f, :2] - y_true_grid_enc[wrong_idx_f, :2],
        #                    axis=1))

        mask_grid = y_pred[:, 4] == y_true_grid_enc[:, 4]

        pos_error_correct_grid, pos_error_wrong_grid, correct_idx_g, wrong_idx_g = compute_acc_of_hit_and_no_hit(
            mask_grid, y_pred, y_true_grid_enc)

        # correct_idx_g = np.where(mask_grid)[0]
        # wrong_idx_g = np.where(~mask_grid)[0]
        #
        # pos_error_correct_grid = np.mean(
        #     np.linalg.norm(
        #         y_pred[correct_idx_g, :2] - y_true_grid_enc[correct_idx_g, :2],
        #         axis=1))
        #
        # pos_error_wrong_grid = np.mean(
        #     np.linalg.norm(y_pred[wrong_idx_g, :2] - y_true_grid_enc[wrong_idx_g, :2],
        #                    axis=1))

        # obtain augmented grid cell ids
        y_gc_aug = self.get_data(self.multi_grid_cell_labels, 'test')

        aug_hit_idx = []
        no_aug_hit_idx = []
        for w_idx in wrong_idx_g:
            aug_idx = np.where(y_gc_aug[w_idx, :] > 0.0)[0]
            if y_pred[w_idx, 4] in aug_idx.tolist():
                aug_hit_idx.append(w_idx)
            else:
                no_aug_hit_idx.append(w_idx)

        aug_hit_idx = np.array(aug_hit_idx)
        no_aug_hit_idx = np.array(no_aug_hit_idx)

        if len(aug_hit_idx) > 0:
            # pos error aug_hit
            pos_error_aug_hit = np.mean(
                np.linalg.norm(
                    y_pred[aug_hit_idx, :2] - y_true_grid_enc[aug_hit_idx, :2],
                    axis=1))
        else:
            pos_error_aug_hit = -1

        if len(no_aug_hit_idx) > 0:
            pos_error_no_aug_hit = np.mean(
                np.linalg.norm(
                    y_pred[no_aug_hit_idx, :2] - y_true_grid_enc[no_aug_hit_idx, :2],
                    axis=1))
        else:
            pos_error_no_aug_hit = -1

        if len(wrong_idx_g) > 0:
            aug_hit_ratio = len(aug_hit_idx) / len(wrong_idx_g)
        else:
            aug_hit_ratio = -1

        metrics['floor_ACC'] = len(floor_acc) / len(floor_true)
        metrics['aug_hit_ratio'] = aug_hit_ratio
        metrics['grid_or_aug_hit_ratio'] = (len(correct_idx_g) + len(aug_hit_idx)) / len(floor_true)
        metrics['grid_ACC'] = len(grid_cell_acc) / len(floor_true)
        metrics['MSE'] = pos_error
        metrics['MSE (correct-floor)'] = pos_error_correct_floor
        metrics['MSE (wrong-floor)'] = pos_error_wrong_floor
        metrics['MSE (correct-grid)'] = pos_error_correct_grid
        metrics['MSE (aug_hit)'] = pos_error_aug_hit
        metrics['MSE (no_aug_hit)'] = pos_error_no_aug_hit
        metrics['MSE (wrong-grid)'] = pos_error_wrong_grid
        metrics['MSE (median)'] = pos_error_median

        if compute_error_vec:
            metrics['error_vec'] = error_vec

        return metrics, y_pred[:, :2], floor_pred


def compute_acc_of_hit_and_no_hit(mask_floor, y_pred, y_true_grid_enc):
    correct_idx_f = np.where(mask_floor)[0]
    wrong_idx_f = np.where(~mask_floor)[0]

    pos_error_correct = np.mean(
        np.linalg.norm(
            y_pred[correct_idx_f, :2] - y_true_grid_enc[correct_idx_f, :2],
            axis=1))

    pos_error_wrong = np.mean(
        np.linalg.norm(y_pred[wrong_idx_f, :2] - y_true_grid_enc[wrong_idx_f, :2],
                       axis=1))

    return pos_error_correct, pos_error_wrong, correct_idx_f, wrong_idx_f


def compute_grid_cell_origins_of_encoding(width, height, grid_size):
    # Determine #row,col
    num_row_l1 = np.ceil(height / grid_size) + 1
    num_col_l1 = np.ceil(width / grid_size) + 1
    num = int(num_row_l1 * num_col_l1)

    origins = []

    # Determine centers & origins (lower-left)
    for idx in range(num):
        r_idx = int(idx / num_col_l1)
        c_idx = idx % num_col_l1

        x = c_idx * grid_size
        y = r_idx * grid_size

        origins.append(np.array([x, y]))

    return np.array(origins), num
