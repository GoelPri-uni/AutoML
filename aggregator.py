import argparse
from smbo import SequentialModelBasedOptimization as smbo
import example_run_experiment as RS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv') #using the dataset with max anchor 16000
    
    return parser.parse_args()


class Aggregator():

    def __init__(self):
        self.max_anchor = 16000
        self.total_budget = 200
        self.args = parse_args()
        self.smbo_result = None
        self.RS_result = None
        self.SH_result = None

    def call_smbo(self):
        self.smbo_result = smbo.eval_smbo(self.args, max_anchor=self.max_anchor, total_budget=self.total_budget)

    def call_RS(self):
        self.RS_result = RS.main()

    def call_SH(self):
        self.SH_result = None

    def comparison_plot(self):
        return None
    
