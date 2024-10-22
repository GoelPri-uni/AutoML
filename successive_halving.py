import ConfigSpace
import numpy as np
import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import matplotlib.pyplot as plt
from surrogate_model import SurrogateModel
from operator import itemgetter

import math
#can we change the value of 1 in the first question
#what are the names of the datasets

def identify_categorical_numerical( df):
        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            # If the column has an object data type or boolean (which could represent categories), treat as categorical
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_cols.append(col)
            # Otherwise, it's treated as numerical
            else:
                numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances/config_performances_dataset-6.csv')
    
    return parser.parse_args()

def input_preprocessor(X_df):
    
    categorical_cols, numerical_cols = identify_categorical_numerical(X_df)
    
    # Define transformers
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def dynamic_preprocessor(X_df):
    """
    A wrapper for creating the preprocessor based on the data passed to the pipeline.
    """
    preprocessor = input_preprocessor(X_df)
    return preprocessor

def clean_configuration(config_space, theta):
    default_values = {}

    # Get the default value
    
    for key, _ in config_space.items():
        default_values[key] = config_space.get_hyperparameter(key).default_value
    
    
    # Step 2: Find missing keys in the partial configuration
    missing_keys = [key for key in default_values.keys() if key not in theta]
    
    # Step 3: Add missing keys with default values to the configuration
    for key in missing_keys:
        theta[key] = default_values[key]
        
        
    return theta

class SuccessiveHalvingBasedOptimization(object):

    def __init__(self, config_space):
        """
        Initializes empty variables for the model, best performance and configuration , the first set of thetas and considering the total samples as 4000 for experiments
        """
        self.config_space = config_space
        self.best_configuration = None
        self.best_performance = None
        self.result_performance = []
        self.model = None
        
        self.thetas = None
        self.total_samples = 4000
        
        
        
    def initialize(self) -> None:
        """
        Initializes the model by training on the entire set of datasets with all the anchor sizes. 
        
        """
        df_data  = pd.read_csv(args.configurations_performance_file)
        surrogate_model = SurrogateModel(self.config_space)
        
        self.model = surrogate_model.fit(df_data)
        print("Trained the model across all anchor sizes")
        anchor_sizes = df_data['anchor_size'].unique()
        self.anchor_sizes = sorted(anchor_sizes)
        
        
   
    def create_initial_configurations(self) -> ConfigSpace.Configuration:
        """

        :return: thetas: A size n vector, containing all configurations handling any missing columns from sample configs
        """
        
        #based on the sample space, finding the  iterations after which only one config will be remaining
        
        
        self.total_iterations = len(bin(self.total_samples)[2:])
        
        
        candidates = [dict(self.config_space.sample_configuration()) for _ in range(self.total_samples)]  # You can adjust the number of samples
        
        #add default values in missing columns from sample configurations
        
        candidates = [clean_configuration(self.config_space, each_candidate) for each_candidate in candidates]
    
        candidate_configs = pd.DataFrame([candidate for candidate in candidates])
        
        thetas = candidate_configs.iloc[::]  
        return thetas


    def pruning(self, performances):
        """
        get the top half best performances and return the positions
        """
        sorted_indices = np.argsort(performances)[:len(performances)//2]
        return sorted_indices
        
    
    
    def select_evenly_spaced_elements(self, lst, n):
        step_size = (len(lst) - 1) / (n - 1)
        return [round(i * step_size) for i in range(n)]
     
    
    def get_anchors(self,full_training_size, total_iterations, threshold):
        n = full_training_size
        halved_values = [full_training_size]
        iteration = 1
        while iteration <=total_iterations and int(n//2) > threshold:
            
            n = int(n // 2)
            halved_values.append(n)
            iteration +=1
        return halved_values[::-1]
   
    def prune_update_optimize(self, thetas):
        """
        Finalize the anchor sizes to experiment considering the full training set size. 
        After evaluation of the configuration using the model, find the top half of the perofrmances with best scores (in our case minimum error rates).
        Select the batch for next iteration with an increased anchor size

        :param thetas: all the initial set of sample configurations 
        
        """
        
         
        #list of anchor sizes for the given iterations including the full training set size
        
        self.possible_anchors = self.get_anchors(max(self.anchor_sizes), self.total_iterations, min(self.anchor_sizes))
        
    
        #self.all_anchor_sizes = np.linspace(min(self.anchor_sizes), max(self.anchor_sizes), self.total_iterations-1, dtype=int)
                 
        #self.all_anchor_sizes = [500,1000,2000,4000,8000,16000]
        
        iteration = 0

        current_performances_index = np.full(len(thetas), np.nan)
        thetas1 = thetas.copy()
        
        all_performances = []
        anchors_travelled = []
        
    
        
        for anchor_ in self.possible_anchors:
            
            thetas1['anchor_size'] = anchor_
            performances = self.model.predict(thetas1)
           
            if iteration == 0:
                current_performances_index = performances
               
            
            positions = self.pruning(performances)
            sorted_performances = performances[positions]
            
            thetas1 = thetas1.iloc[positions]
            
            current_index_list = thetas1.index.to_list()
            current_performances_index = np.full(len(thetas), np.nan)
            current_performances_index[current_index_list] = sorted_performances
            all_performances.append(current_performances_index)
            
            

            anchors_travelled.append(anchor_)
            if len(self.possible_anchors)-1 == iteration:
                
                self.best_configuration = thetas1.iloc[0]
                self.best_performance = sorted_performances[0]
                self.best_anchor = anchor_
                return all_performances,anchors_travelled
            
            iteration +=1
        
   
    
    def plot_successive_halving(self , all_performances, anchors_travelled):

        #prepare to plot
        plt.figure(figsize=(10, 6))
        
        df = pd.DataFrame(all_performances)
        
        #df = pd.DataFrame(all_performances)
        
        #giving names to create columns for dataframe to plot the values wrt performances and anchor
        df.columns = [f"Config_{i+1}" for i in range(len(all_performances[0]))]
        
        plt.figure(figsize=(10, 6))

        #plot the performances
        for config in df.columns:  
            plt.plot(df.index, df[config], marker='o', linestyle='-')
            

        # Adding labels, title, and legend
        plt.xlabel('Anchors in Successive Halving)')
        plt.ylabel('Performance')
        plt.title('Successive Halving: Performance Progression')
        plt.xticks(ticks= list(range(len(anchors_travelled))),labels= anchors_travelled)  # Set iteration points on the x-axis
        
        yticks = list(plt.yticks()[0])
        
        if self.best_performance not in yticks:
            yticks.append(self.best_performance)
            yticks = sorted(yticks)  # Sort the ticks to keep them in order
            
        plt.axhline(y=self.best_performance, color='red', linestyle='--', label=f'Best Score: {self.best_performance:.6f}')
        

            # Add legend to show which line corresponds to which configuration
        plt.yticks(yticks)
        
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
def eval_successive(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    config_space.seed(42)
    s_h_method =  SuccessiveHalvingBasedOptimization(config_space=config_space)
    
    s_h_method.initialize()
    thetas = s_h_method.create_initial_configurations()
    all_performances, anchors_travelled = s_h_method.prune_update_optimize(thetas)
    
    print(s_h_method.best_configuration)
    print(s_h_method.best_performance)
    
    s_h_method.plot_successive_halving(all_performances, anchors_travelled)
    return s_h_method
    
if __name__ == '__main__':
    args = parse_args()
    
    s_h_method = eval_successive(args)
    print(s_h_method.best_configuration)
    print(s_h_method.best_performance)
    
    # config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    # config_space.seed(42)
    # s_h_method =  SuccessiveHalvingBasedOptimization(config_space=config_space)
    
    # s_h_method.initialize()
    # thetas = s_h_method.create_initial_configurations()
    # all_performances, anchors_travelled = s_h_method.prune_update_optimize(thetas)
    
    # print(s_h_method.best_configuration)
    # print(s_h_method.best_performance)
    
    # s_h_method.plot_successive_halving(all_performances, anchors_travelled)
     