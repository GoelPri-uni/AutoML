
import ConfigSpace
import pandas as pd
from ConfigSpace import UniformIntegerHyperparameter, Constant
from surrogate_model import SurrogateModel
import pickle
import argparse

#add our own anchor
#fillna as 0 in datframe
#can i create helper file - joint functions throughout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    # max_anchor_size: connected to the configurations_perfor  mance_file. The max value upon which anchors are sampled
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    return parser.parse_args()




def test_external_model(config_space, model_path, data):
    external_model = load_model(model_path)
    sg = SurrogateModel(config_space=config_space)

    sg.model = external_model
    sg.config_space = config_space
    theta_val = dict(data)
    
    error_rate = sg.predict(theta_val)
    return error_rate

def get_initial_sample_config(config_space, model_path):
        capital_phi = []
        external_model = load_model(model_path)
        
        
        for _ in range(10):  # Sample 10 initial configurations
            config = config_space.sample_configuration()
            sg = SurrogateModel(config_space=config_space)
           
            sg.model = external_model
            sg.config_space = config_space
            theta_val = dict(config)
            error_rate = sg.predict(theta_val)
            capital_phi.append((theta_val, error_rate))
        return capital_phi

def load_model(filename):
    """ Load the model from a file using pickle """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

def add_anchor(config_space):
    anchor_size = UniformIntegerHyperparameter("anchor_size", lower=16, upper=1600)
    
    config_space.add_hyperparameter(anchor_size)
    return config_space

def initial_perf(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)

    # Define other hyperparameters...

    # Define anchor_size as a hyperparameter
    anchor_size =  Constant("anchor_size", 1600)
    
    config_space.add_hyperparameter(anchor_size)
    model_path = args.model_path
    capital_phi = get_initial_sample_config(config_space, model_path)
    
    #surrogate_model.fit(df)
    return capital_phi
if __name__ == '__main__':
    
    captial_phi = initial_perf(parse_args())
 