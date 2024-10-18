import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel

"""
a.
    1. external surrogate model that is trained on all anchor sizes -i.e., we fit the entire dataset (to use its predictions as ground truth)
b.
    2. initialize configurations -> sampled first (with a certain anchor size, start at 256?)
        - anchor sizes: [  16   23   32   45   64   91  128  181  256  362  512  724 1024 1448 1600]
    3. evaluate configs
    4. choose top half configs (with best performance) & discard the rest
    5. increase anchor size 
    6. repeat until we ran out of anchor sizes 
"""



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    #successive halving parameters
    parser.add_argument('--R', type=int, default=[256, 512, 1024, 1600]) #maximum resource / biggest anchor size
    parser.add_argument('--eta', type=int, default=2) #halving parameter


    return parser.parse_args()


def plot_halving(config_list):
    '''
    config_list -> nested list where each inner list contains dictionaries of the configs stored at each
    anchor size
    '''
    for lst in config_list:
        print(len(lst))
        for config in lst:
            try:
                print(config['score'])
            except KeyError:
                print('no score available')

    return


def prune_worst(config_list, args):
    
    #1. sorting the configurations in ascending order (error rate)
    sorted_configs =  sorted(config_list, key=lambda x: x['score']) 

    #for config in sorted_configs:
        #print(config['score'])

    #2. only keeping the first half of the configs
    to_keep = len(sorted_configs) // args.eta 
    halved_configs = sorted_configs[:to_keep]
    #print('halved configs', halved_configs)

    #3. remove the score keys before returning the configs

    for config_dict in halved_configs:
        config_dict = config_dict.pop('score')
  
    return halved_configs


def run(args):
    
    # Select all configurations per anchor size
    #anchor_sizes = [256, 512, 1024, 1600]

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file) 
    config_samples = config_space.sample_configuration(20) #sample configurations to evaluate
    
    df = pd.read_csv(args.configurations_performance_file)
    
    external_surrogate = SurrogateModel(config_space)
    external_surrogate.fit(df)

    config_log = [] #storing only the current configs and their scores 
    plot_list = [] #storing the configs (and their scores) evaluated at each anchor size

    for anchor in args.R: #iteratively increase anchor size 
        # get the performance of each sampled config
        config_log = [] # clean out the previous configs
        for config in config_samples: 
            #only the first time do the configs need to be transformed into a dict (from Config)
            if anchor == args.R[0]: #if we're dealing with the first anchor, samples need to be initialized from config space
                theta_new = dict(config)
            else:
                theta_new = config

            theta_new['anchor_size'] = anchor
            #print('theta_new', theta_new)

            performance = external_surrogate.predict(theta_new)
            theta_new['score'] = performance
            config_log.append(theta_new)

        #print('results', config_log)
        print('one bracket done, anchor size: ', anchor)

        # store configs and their scores for plotting
        plot_list.append(config_log) 
        
        # prune the worst performing configs
        best_configs = prune_worst(config_log, args)
        
        # only use the remaining configs for the next anchor size
        config_samples = best_configs
    
    plot_halving(plot_list)

    return 



if __name__ == '__main__':
    run(parse_args())
