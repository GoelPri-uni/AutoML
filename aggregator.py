import argparse
import smbo
import example_run_experiment as RS
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv') #using the dataset with max anchor 16000
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=200)

    return parser.parse_args()


class Aggregator():
    '''
    An aggregator class that pulls together the results
    of Random Search, SMBO and Successive Halving for comparison plots.
    '''

    def __init__(self):
        self.args = parse_args()
        self.smbo_result_mean = []
        self.RS_result_mean = []
        self.SH_result_mean = []

    def call_smbo(self, idx):
        for x in range(0,idx):
            smbo_result = smbo.eval_smbo(self.args, max_anchor=self.args.max_anchor_size, total_budget=self.args.num_iterations)
            self.smbo_result_mean.append(np.array(smbo_result))
        #print('smbo result: ', self.smbo_result, len(self.smbo_result))

    def call_RS(self, idx):
        for x in range(0,idx):
            RS_result = RS.run(self.args, aggregator_request=True, test_dataset=self.args.configurations_performance_file)
            self.RS_result_mean.append(np.array(RS_result['random_search']))


    def call_SH(self):
        self.SH_result = None

    def get_means_and_scores(self):
        #get a list of mean values over n iterations for each algorithm 
        mean_values_RS = [np.mean(scores) for scores in zip(*self.RS_result_mean)]
        mean_values_smbo = [np.mean(scores) for scores in zip(*self.smbo_result_mean)]

        #get variance for each mean 
        std_RS = [np.std(scores) for scores in zip(*self.RS_result_mean)]
        std_smbo = [np.std(scores) for scores in zip(*self.smbo_result_mean)]

        #get the best score and iteration index for each algorithm
        best_RS = min(mean_values_RS)
        best_smbo = min(mean_values_smbo)
        iteration_RS = mean_values_RS.index(best_RS)
        iteration_smbo = mean_values_smbo.index(best_smbo)

        #return mean values, variance, best scores, iteration index
        return mean_values_RS, mean_values_smbo, std_RS, std_smbo, best_RS, best_smbo, iteration_RS, iteration_smbo

    def plot_smbo_RS(self):

        '''
        Places the results of smbo and Random Search in one plot.
        '''

        mean_values_RS, mean_values_smbo, std_RS, std_smbo, best_RS, best_smbo, iteration_RS, iteration_smbo = self.get_means_and_scores()

        '''

         #Get the smallest score that was found
        smallest_score = min(self.RS_result['random_search'])
        # Find the earliest index of the best score
        earliest_index = self.RS_result['random_search'].index(smallest_score) #the first value in the list does not count as an iteration
        print('score and iter ', smallest_score, earliest_index)
        # Get the current y-ticks
        #yticks = np.linspace(min(self.RS_result['random_search']), max(self.RS_result['random_search'][1:]), num=6) #list(set(results['random_search'][1:]))

        #print('yticks ', yticks)
        '''

        plt.plot(range(len(mean_values_RS)-1), mean_values_RS[1:], label='Random Search mean scores')
        plt.scatter(iteration_RS, best_RS, color='red') #zorder=5, label=' Best score='+str(round(smallest_score,3))+", \n iteration="+str(earliest_index)) 
        plt.fill_between(range(len(mean_values_RS)-1), (np.array(mean_values_RS[1:]) + np.array(std_RS[1:])), (np.array(mean_values_RS[1:]) - np.array(std_RS[1:])), alpha=0.3)

        plt.plot(range(len(mean_values_smbo)), mean_values_smbo, label='SMBO mean scores')
        plt.scatter(iteration_smbo, best_smbo, color='red')     
        plt.fill_between(range(len(mean_values_smbo)), (np.array(mean_values_smbo) + np.array(std_smbo)), (np.array(mean_values_smbo) - np.array(std_smbo)), alpha=0.3)
   
        #plt.ylim(0,1)
        #plt.yticks(yticks)
        #plt.yscale('log')
        plt.xlabel('Num iterations')
        plt.ylabel('Score')
        plt.title('Random Search and SMBO performance on the Letter dataset \n Max anchor size: ' + str(self.args.max_anchor_size))
        #plt.annotate('Spearman rho='+str(round(spearman_rho,3)), [320, yticks[-3]])
        #plt.axhline(y=min(results['random_search']), color='red', linestyle='--', label='Best score')
        plt.legend()
        plt.tight_layout()
        plt.show()

        
    
    def plot_smbo_SH(self):
        '''
        Plots the results of SMBO and successive halving in one plot.
        '''
        return NotImplementedError

if __name__=='__main__':
    agg = Aggregator()
    agg.call_smbo(4)
    agg.call_RS(4)
    agg.plot_smbo_RS()
    print('done')
    
