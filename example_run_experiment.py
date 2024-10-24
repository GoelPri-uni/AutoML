import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=list, default= ['lcdb_configs.csv', 'config_performances_dataset-6.csv', 'config_performances_dataset-11.csv', 'config_performances_dataset-1457.csv'])

    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()



def run(args, aggregator_request=False, test_dataset='config_performances_dataset-6.csv'):
    #iterate over the different datasets 
    dataset_names = ['Learning Curve Database', 'Letter', 'Balance-scale', 'Amazon commerce review']

    if aggregator_request:
        dataset_lst = [test_dataset]
        num_iterations = 200
    else:
        dataset_lst = args.configurations_performance_file
        num_iterations= args.num_iterations

    for dataset in dataset_lst:
        
        config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
        #config_space.seed(42) #fixing the random seed for plot reproducibility
        random_search = RandomSearch(config_space)
        
        df = pd.read_csv(dataset) #(args.configurations_performance_file)
        surrogate_model = SurrogateModel(config_space)
        
        max_anchor_size = max(df['anchor_size'].unique()) #find out the max anchor size for each dataset
        print('max anchor size', max_anchor_size)
        surrogate_model.fit(df[df['anchor_size'] == max_anchor_size]) #fit the surrogate model only on the configurations that have the max anchor size
        
        results = {
            'random_search': [1.0]   
        }

        for idx in range(num_iterations):
            theta_new = dict(random_search.select_configuration())
            
            theta_new['anchor_size'] = max_anchor_size
            performance = surrogate_model.predict(theta_new)
            # ensure to only record improvements
        
            results['random_search'].append(min(results['random_search'][-1], performance))
            random_search.update_runs((theta_new, performance))

        
        if aggregator_request:
            return results
        
        #Get the smallest score that was found
        smallest_score = min(results['random_search'])
        # Find the earliest index of the best score
        earliest_index = results['random_search'].index(smallest_score) #the first value in the list does not count as an iteration
        
        dataset_idx = args.configurations_performance_file.index(str(dataset)) #get dataset name
        spearman_rho = surrogate_model.get_spearman() #get the spearman correlation
        
        
        # Get the current y-ticks
        yticks = np.linspace(min(results['random_search']), max(results['random_search'][1:]), num=6) #list(set(results['random_search'][1:]))

        print('yticks ', yticks)

        plt.plot(range(len(results['random_search'])-1), results['random_search'][1:]) #leaving out the initial 1.0 value from the plot, only plotting improvements
        plt.yticks(yticks)
        #plt.yscale('log')
        plt.xlabel('Num iterations')
        plt.ylabel('Score')
        plt.title('Surrogate model score predictions on ' + str(dataset_names[dataset_idx]) + '\n Max anchor size: ' + str(max_anchor_size))
        plt.scatter(earliest_index, smallest_score, color='red', zorder=5, label=' Best score='+str(round(smallest_score,3))+", \n iteration="+str(earliest_index)) #+ "\n Spearman's rho="+str(round(spearman_rho,3)))
       
        if aggregator_request:
            plt.annotate('Spearman rho='+str(round(spearman_rho,3)), [120, yticks[-3]])
        else:
            plt.annotate('Spearman rho='+str(round(spearman_rho,3)), [320, yticks[-3]])

        plt.axhline(y=min(results['random_search']), color='red', linestyle='--', label='Best score')
        plt.legend()
        plt.tight_layout()
        plt.show()
        



if __name__ == '__main__':
    run(parse_args())
