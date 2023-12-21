from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import time

@dataclass
class Visualizer:
    pred_type: str
    fairness_matrix: str
    out_dir: str
    attr_list: list[str] | None = None
    marker_list: list[str] = field(default_factory=lambda: ['o', '^', '*', 'd', 'p', 'x'])
    markersize: int = 10
    color_list: list[str] = field(default_factory=lambda: ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])
    
    def resolve_fairness(self, stats):
        def get_binary_stats_per_epoch(stats_per_epoch):
            # the stats should have shape [A, 8]
            group_1_tp, group_1_fp, group_1_fn, group_1_tn = [stats_per_epoch[:,i] for i in range(0, 4)]
            group_2_tp, group_2_fp, group_2_fn, group_2_tn = [stats_per_epoch[:,i] for i in range(4, 8)]
            # Accuracy
            total_accuracy = (group_1_tp+group_1_tn+group_2_tp+group_2_tn)/(group_1_tp+group_1_fp+group_1_fn+group_1_tn+group_2_tp+group_2_fp+group_2_fn+group_2_tn)
            group_1_accuracy = (group_1_tp+group_1_tn)/(group_1_tp+group_1_fp+group_1_fn+group_1_tn)
            group_2_accuracy = (group_2_tp+group_2_tn)/(group_2_tp+group_2_fp+group_2_fn+group_2_tn)
            # Fairness
            group_1_tpr = group_1_tp/(group_1_tp+group_1_fn)
            group_2_tpr = group_2_tp/(group_2_tp+group_2_fn)
            group_1_tnr = group_1_tn/(group_1_fp+group_1_tn)
            group_2_tnr = group_2_tn/(group_2_fp+group_2_tn)
            equality_of_opportunity = abs(group_1_tpr-group_2_tpr) # (0, 1)
            equality_of_opportunity[np.isnan(equality_of_opportunity)] = 0
            equalized_odds = abs(group_1_tpr-group_2_tpr) + abs(group_1_tnr-group_2_tnr) # (0, 2)
            equalized_odds[np.isnan(equalized_odds)] = 0
            return {'total_accuracy': np.expand_dims(total_accuracy, axis=0),
                    'group_1_accuracy': np.expand_dims(group_1_accuracy, axis=0),
                    'group_2_accuracy': np.expand_dims(group_2_accuracy, axis=0),
                    'equality_of_opportunity': np.expand_dims(equality_of_opportunity, axis=0),
                    'equalized_odds': np.expand_dims(equalized_odds, axis=0)}
        def get_categorical_stats_per_epoch(stats_per_epoch):
            # the stats should have shape [A, 4]
            group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [stats_per_epoch[:,i] for i in range(0, 4)]
            group_1_acc = group_1_correct/(group_1_correct+group_1_wrong)
            group_2_acc = group_2_correct/(group_2_correct+group_2_wrong)
            total_acc = (group_1_correct+group_2_correct)/(group_1_correct+group_1_wrong+group_2_correct+group_2_wrong)
            accuracy_difference = abs(group_1_acc-group_2_acc)
            return {"group_1_accuracy": np.expand_dims(group_1_acc, axis=0), "group_2_accuracy": np.expand_dims(group_2_acc, axis=0), 
                    "total_accuracy": np.expand_dims(total_acc, axis=0), "accuracy_difference": np.expand_dims(accuracy_difference, axis=0)}
        match self.pred_type:
            case "binary":
                # the stats should have shape [N, A, 8]
                stats_dict = {'total_accuracy': np.array([]), 'group_1_accuracy': np.array([]), 'group_2_accuracy': np.array([]),
                              'equality_of_opportunity': np.array([]), 'equalized_odds': np.array([])}
                for stats_per_epoch in stats:
                    d = get_binary_stats_per_epoch(stats_per_epoch)
                    for key, values in stats_dict.items():
                        stats_dict.update({key: np.concatenate((values, d[key]), axis=0) if values.size > 0 else d[key]})
            case "categorical":
                # the stats should have shape [N, A, 4]
                stats_dict = {'total_accuracy': np.array([]), 'group_1_accuracy': np.array([]), 'group_2_accuracy': np.array([]), 'accuracy_difference': np.array([])}
                for stats_per_epoch in stats:
                    d = get_categorical_stats_per_epoch(stats_per_epoch)
                    for key, values in stats_dict.items():
                        stats_dict.update({key: np.concatenate((values, d[key]), axis=0) if values.size > 0 else d[key]})
            case _:
                assert False, "unknown prediction types"
        return stats_dict
    
    def draw_status_per_attribute(self, stats_list, legend_list, length=None, description=None):
        def to_dict_key(str):
            match str:
                case "equality of opportunity":
                    assert self.pred_type == "binary", f'prediction type and fairness matrix mismatch'
                    return "equality_of_opportunity"
                case "equalized odds":
                    assert self.pred_type == "binary", f'prediction type and fairness matrix mismatch'
                    return "equalized_odds"
                case "accuracy difference":
                    assert self.pred_type == "categorical", f'prediction type and fairness matrix mismatch'
                    return "accuracy_difference"
                case _:
                    assert False, f'unrecognized fairness criteria'
        # one image per attribute
        for attr_idx, attr_name in enumerate(self.attr_list):
            fig_root = Path(self.out_dir)
            fig_root.mkdir(parents=True, exist_ok=True)
            fig_path = fig_root / (attr_name + '.png')
            # get all stats in the stat list
            stats_dict_list = list() # a list with dict as its element
            for stat in stats_list:
                stats_dict_list.append(self.resolve_fairness(stat))
            # draw chart
            fig, axs  = plt.subplots(1,1, figsize=(8,8),)
            epoch_axis = np.linspace(0, length-1, length) if length else np.linspace(0, stats_list[0].shape[0]-1, stats_list[0].shape[0])
            for idx, stats_dict in enumerate(stats_dict_list):
                if idx == 0:  # ignore direct
                    continue
                # line2d object ()
                line = mlines.Line2D(list(map(lambda x: 1.0-x, stats_dict[to_dict_key(self.fairness_matrix)][:epoch_axis.shape[0],attr_idx])), 
                               stats_dict['total_accuracy'][:epoch_axis.shape[0],attr_idx], 
                               marker=self.marker_list[idx], markersize=self.markersize, color=self.color_list[idx], fillstyle='none', linestyle=" ") # linestyle=" " for no line
                axs.add_line(line)

                # line, = axs.plot(list(map(lambda x: 1.0-x, stats_dict[to_dict_key(self.fairness_matrix)][:epoch_axis.shape[0],attr_idx])), 
                #                stats_dict['total_accuracy'][:epoch_axis.shape[0],attr_idx], marker=self.marker, markersize=self.markersize)
                
            # axs.set_xlabel(f'{self.fairness_matrix}')
            axs.set_xlabel(f'Fairness (higher the better)\n{description[attr_idx]}', fontsize="18")
            axs.set_ylabel('Total Accuracy (higher the better)', fontsize="18",)
            axs.legend(legend_list[1:], fontsize="20", loc="lower left") # ignore direct
            # axs.legend(legend_list, fontsize="20", loc="lower left")
            axs.set_box_aspect(1)
            axs.autoscale()
            fig.tight_layout()
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            
def main(args):

    time_start = time.perf_counter()
    # load all the stats
    stats_path_list = [Path(p) for p in args.stats]
    stats_list = [np.load(p) for p in stats_path_list]
    assert len(stats_list) == len(args.legends) , "number of stats not equal to legends"
    print(f'load {len(stats_list)} stats...')
    visualizer = Visualizer(pred_type=args.pred_type, fairness_matrix=args.fairness_matrix, out_dir=args.out_dir, attr_list=args.attr_list)
    visualizer.draw_status_per_attribute(stats_list=stats_list, legend_list=args.legends, description=args.descriptions)

    time_end = time.perf_counter()
    print(f'done in {(time_end-time_start)/60:.4f} mins')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Show comparison between different method")

    parser.add_argument("--stats", type=str, nargs='+', help="path to the validation stats, should be a .npy file")
    parser.add_argument("--legends", type=str, nargs='+', help="name for the stats, should be the same number as the stats")
    parser.add_argument("--descriptions", type=str, nargs='+', help="words to put on the bottom of the graph")
    parser.add_argument("-l", "--length", type=int, default=None, help="number of epochs shown in graph, start from 0")

    parser.add_argument("--pred-type", type=str, help="model prediction type, binary or categorical, saved in different format")
    parser.add_argument("--fairness-matrix", default="accuracy difference", help="how to measure fairness")
    parser.add_argument("--attr-list", type=str, nargs='+', help="attributes name predicted by model")


    parser.add_argument("-o", "--out-dir", type=str, help="output folder for all graph output")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)