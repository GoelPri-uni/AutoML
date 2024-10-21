import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=list, default= ['lcdb_configs.csv', 'config_performances_dataset-6.csv', 'config_performances_dataset-11.csv', 'config_performances_dataset-1457.csv'])

    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()


def run(args):
    #iterate over the different datasets 
    dataset_names = ['Learning Curve Database', 'Letter', 'Balance-scale', 'Amazon commerce review']
    for dataset in args.configurations_performance_file:
        
        config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
        config_space.seed(42) #fixing the random seed for plot reproducibility
        random_search = RandomSearch(config_space)
        
        df = pd.read_csv(dataset) #(args.configurations_performance_file)
        surrogate_model = SurrogateModel(config_space)
        #surrogate_model.fit(df)
        max_anchor_size = max(df['anchor_size'].unique()) #find out the max anchor size for each dataset
        print('max anchor size', max_anchor_size)
        surrogate_model.fit(df[df['anchor_size'] == max_anchor_size]) #fit the surrogate model only on the configurations that have the max anchor size
        
        results = {
            'random_search': [1.0]   
        }

        for idx in range(args.num_iterations):
            theta_new = dict(random_search.select_configuration())
            
            theta_new['anchor_size'] = args.max_anchor_size
            performance = surrogate_model.predict(theta_new)
            # ensure to only record improvements
        
            results['random_search'].append(min(results['random_search'][-1], performance))
            random_search.update_runs((theta_new, performance))

        smallest_score = min(results['random_search'])
        # Get the last number in the list
        first_iter_found = results['random_search'][-1]
        # Find the earliest index of the best score
        earliest_index = results['random_search'].index(first_iter_found) #the first value in the list does not count as an iteration
        
        #dataset_name = dataset_names[i]
        dataset_idx = args.configurations_performance_file.index(str(dataset))
        spearman_rho = surrogate_model.get_spearman()
        #custom_legend = [plt.Line2D([0], [0], color='none', label="Spearman's rho="+str(spearman_rho))]
    

        print('retrieved rho: ', spearman_rho)

        plt.plot(range(len(results['random_search'])), results['random_search'])
        plt.yscale('log')
        plt.xlabel('Num iterations')
        plt.ylabel('Score')
        plt.title('Surrogate model score predictions on ' + str(dataset_names[dataset_idx]) + '\n Max anchor size: ' + str(max_anchor_size))
        plt.scatter(earliest_index, smallest_score, color='red', zorder=5, label=' Best score='+str(round(smallest_score,3))+", \n iteration="+str(earliest_index)) #+ "\n Spearman's rho="+str(round(spearman_rho,3)))
        plt.text(312, 0.7, 'Spearman correlation='+str(round(spearman_rho,3)), fontsize=9.5, bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()
        plt.show()
        

if __name__ == '__main__':
    run(parse_args())
