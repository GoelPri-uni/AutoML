import ConfigSpace
from ConfigSpace import Constant
import numpy as np
import typing
import random
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from example_smbo import initial_perf, test_external_model
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
from random_search import RandomSearch
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from surrogate_model import SurrogateModel

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
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    return parser.parse_args()


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


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space, max_anchor, ex_surrogate_class):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space
        self.R = None
        self.theta_inc =None
        self.theta_inc_performance = None
        self.internal_surrogate_model = None
        self.result_performance = []
        self.random_search = RandomSearch(self.config_space)
        self.max_anchor = max_anchor
        self.all_performances = []
        self.ex_surrogate_class = ex_surrogate_class
    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        """
        receive sample configuration and respective performance using the ConfigSpace library for a given configuration 
        
        """
        self.R = capital_phi
        
        print("initilised the model")
        
        best_value = min(capital_phi,key = lambda x: x[1])
        
        self.theta_inc = best_value[0]  #initialise these values based on the first set of configuration
        self.theta_inc_performance = best_value[1] 
        
  
            
    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        if len(self.R) > 0:
            
            #scaling of data
            configurations = [item[0] for item in self.R]
            X_df = pd.DataFrame(configurations)
        
            categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
            numerical_transformer = StandardScaler()
            categorical_cols, numerical_cols = identify_categorical_numerical(X_df)
            preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
            )
            
            #prepare for training
            X= X_df.iloc[::]
            
            
            y = np.array([performance for _, performance in self.R])  # Error rates
            
            # Define a Gaussian Process model
            kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e1), nu=2.5)

            # Initialize GaussianProcessRegressor with the Matern kernel
            internal_surrogate_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)  # Alpha is noise level
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', internal_surrogate_model)
            ])
            
            self.internal_surrogate_model = pipeline.fit(X, y)
      
            # Fit the model to the observations
            
            return self.internal_surrogate_model

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        
        
        candidates = [self.config_space.sample_configuration().get_dictionary() for _ in range(20)]  
        
        candidates = [clean_configuration(self.config_space, each_candidate) for each_candidate in candidates]
        
        candidate_configs = pd.DataFrame([candidate for candidate in candidates])
        
        theta = candidate_configs.iloc[::] 
        
        # Calculate the expected improvement for each candidate
        ei_values = self.expected_improvement(self.internal_surrogate_model, self.theta_inc_performance, theta)
        
        # Select the configuration with the highest expected improvement
        best_candidate_idx = np.argmax(ei_values)
        best_candidate = candidates[best_candidate_idx]
        return best_candidate

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        
        so, select configuration uses this function to get EI 
        """
        # Predict the mean and standard deviation for the candidates
        
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        if f_star == None:
            f_star = mu
        xi = 0.03  # Exploration parameter
        f_star_exploration = f_star + xi
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        
        sigma_noise = np.maximum(sigma, 1e-2)
        # Calculate the Expected Improvement (EI)
        z = (f_star_exploration - mu) / sigma_noise
        
        ei = (f_star_exploration - mu) * norm.cdf(z) + sigma_noise * norm.pdf(z)
        
        return ei
        
    def optimize(self,  theta_new):
        #performance = test_real_world_model(self.config_space, theta_new)
        
        performance = self.ex_surrogate_class.external_surrogate_predict(theta_new)
        return performance
    
        
    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        
        #adding random candidate to R for new exploration area to avoid stagnant behaviour
        random_candidate = self.config_space.sample_configuration().get_dictionary()  
        
        candidate = clean_configuration(self.config_space, random_candidate)
        
        random_cand_perf = smbo.optimize(candidate)
        
        self.R.append((candidate, random_cand_perf))
        
        self.R.append(run)
        
        
        if run[1] < self.theta_inc_performance:
            self.theta_inc_performance = run[1]
            self.theta_inc = run[0]
        
        self.result_performance.append(self.theta_inc_performance)

import pickle

class ExternalSurrogate():
    def __init__(self, args):
        model_path = args.model_path
        external_model = self.load_model(model_path)
        self.sg = SurrogateModel(config_space=config_space)
        self.sg.model = external_model
        
    def external_surrogate_predict(self, theta):
        theta_val = dict(theta)
        error_rate = self.sg.predict(theta_val)
        return error_rate
    
    def load_model(self, filename):
        """ Load the model from a file using pickle """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def get_initial_sample_config(self, config_space):
            capital_phi = []
            
            
            for _ in range(20):  # Sample 10 initial configurations
                config = config_space.sample_configuration()
                
                
            
                
                self.sg.config_space = config_space
                theta_val = dict(config)
                error_rate = self.sg.predict(theta_val)
                capital_phi.append((theta_val, error_rate))
                
            return capital_phi






    
if __name__ == '__main__':
    args = parse_args()
    
    
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    
    
    max_anchor = 1600
    
    anchor_size =  Constant("anchor_size", max_anchor)
    
    config_space.add_hyperparameter(anchor_size)
    ex_surrogate_class = ExternalSurrogate(args=args)
    capital_phi = ex_surrogate_class.get_initial_sample_config(config_space)
    
    smbo = SequentialModelBasedOptimization (config_space=config_space, max_anchor=max_anchor, ex_surrogate_class=ex_surrogate_class)
    
    
    smbo.initialize(capital_phi)
    total_budget = 50
    budget_left = 50

 
    while budget_left:
        smbo.fit_model()
        theta_new = smbo.select_configuration() 
        performance = smbo.optimize(theta_new)
        smbo.update_runs((theta_new , performance))
        budget_left = budget_left-1
       
        smbo.all_performances.append(performance)
    
    
    width = max(6, total_budget // 10)  # Dynamically calculate width
    height = 6  # Set a constant height

    # Create a figure with dynamic size
    plt.figure(figsize=(width, height))

    plt.plot(range(total_budget), smbo.result_performance, color='blue', label='Iterative Best Performances so far')
    
    plt.plot(range(total_budget), smbo.all_performances, color='black', label = 'All SMBO performances')
    
    
    min_performance = min(smbo.result_performance)

    min_budget = smbo.result_performance.index(min_performance)

    plt.scatter(min_budget, min_performance, color='red', zorder=5, label=' Best Performance')

    # Add a horizontal line at the minimum performance point
    plt.axhline(y=min_performance, color='red', linestyle='--', label=f'Min: {min_performance:.6f}')
    # Get the current y-ticks
    yticks = list(plt.yticks()[0])

    # Add the minimum performance to the y-ticks if it's not already there
    if min_performance not in yticks:
        yticks.append(min_performance)
        yticks = sorted(yticks)  # Sort the ticks to keep them in order

    # Set the updated y-ticks
    plt.yticks(yticks)
    plt.xlabel('Budget')
    plt.ylabel('Guassian Regressor Performance')
    plt.title('Performances tracked during SMBO')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    