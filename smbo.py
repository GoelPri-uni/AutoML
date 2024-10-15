import ConfigSpace
from ConfigSpace import Constant
import numpy as np
import typing
import random
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from example_smbo import initial_perf, add_anchor, test_external_model
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import math
from scipy.stats import norm

import matplotlib.pyplot as plt


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
    # max_anchor_size: connected to the configurations_perfor  mance_file. The max value upon which anchors are sampled
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
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
    
    
    #active_hyperparameters = config_space.get_active_hyperparameters(theta)
    
    
    # Step 2: Find missing keys in the partial configuration
    missing_keys = [key for key in default_values.keys() if key not in theta]
    
    # Step 3: Add missing keys with default values to the configuration
    for key in missing_keys:
        theta[key] = default_values[key]
        #theta[key] = config_space.get_hyperparameter(key).default_value
        
    return theta

def test_real_world_model(config_space, theta_new):
    model_path = parse_args().model_path
    performance = test_external_model(config_space, model_path, theta_new)  
    return performance

class SequentialModelBasedOptimization(object):

    def __init__(self, config_space):
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
            print(X_df)
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
            
            #X = np.array([list(config.values()) for config, _ in self.R])  # Configurations as feature vectors
            y = np.array([performance for _, performance in self.R])  # Error rates
            
            # Define a Gaussian Process model
            kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
            
            internal_surrogate_model = GaussianProcessRegressor(kernel=kernel)
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', internal_surrogate_model)
            ])
            
            self.internal_surrogate_model = pipeline.fit(X, y)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            #print(trained_model.predict(X_test))
            
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
        
        candidates = [self.config_space.sample_configuration().get_dictionary() for _ in range(10)]  # You can adjust the number of samples
        
        candidates = [clean_configuration(self.config_space, each_candidate) for each_candidate in candidates]
    
        candidate_configs = pd.DataFrame([candidate for candidate in candidates])
        
        theta = candidate_configs.iloc[::]  # Apply the same pre-processing steps as in the model
        
        
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
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        
        # Calculate the Expected Improvement (EI)
        z = (f_star - mu) / sigma
        
        ei = (f_star - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
        
    def optimize(self, theta_new):
        performance = test_real_world_model(self.config_space, theta_new)
        
        return performance
    
        
    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        
        self.R.append(run)
        
        if run[1] < self.theta_inc_performance:
            self.theta_inc_performance = run[1]
            self.theta_inc = run[0]
        self.result_performance.append(self.theta_inc_performance)
        
if __name__ == '__main__':
    args = parse_args()
    capital_phi = initial_perf(args)
    
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    anchor_size =  Constant("anchor_size", 1600)
    
    config_space.add_hyperparameter(anchor_size)
    smbo = SequentialModelBasedOptimization (config_space=config_space)
    
    
    smbo.initialize(capital_phi)
    total_budget = 12
    budget_left = 12
    each_performance = []
    while budget_left:
        print("Budget is", budget_left)
        smbo.fit_model ()
        theta_new = smbo.select_configuration() 
        performance = smbo.optimize(theta_new)
        smbo.update_runs((theta_new , performance))
        budget_left = budget_left-1
        print(performance)
        each_performance.append(performance)
    #plt.plot(range(total_budget), smbo.result_performance)
    #plt.plot(range(total_budget), each_performance)
    
    
    min_performance = min(each_performance)
    min_budget = each_performance.index(min_performance)

    # Plot the performance curve
    plt.plot(range(total_budget), each_performance, label='Performance', marker='o')

    # Mark the minimum performance on the plot
    plt.annotate(f'Min: {min_performance:.4f}', xy=(min_budget, min_performance),
                xytext=(min_budget + 1, min_performance + 0.02), 
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12, color='red')
    plt.xlabel('Budget')
    plt.ylabel('Performance')
    plt.title('SMBO Performance with Minimum Marked')
    plt.legend()
    plt.grid(True)
    plt.show()
    