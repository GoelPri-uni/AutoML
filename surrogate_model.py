import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
import pickle
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
 

class SurrogateModel:

    # def __init__(self, config_space):
    #     self.config_space = config_space
    #     self.df = None
    #     self.model = None

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None
    
    def identify_categorical_numerical(self, df):
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
    
    
    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        df = df
        X = df.iloc[:, :-1]  # All columns except the last one(score)
        y = df.iloc[:, -1] # the last column, the performance values
        
        categorical_cols, numerical_cols = self.identify_categorical_numerical(X)
    
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Step 4: Create a pipeline with preprocessing and the model
        model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Step 5: Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       
        # Step 6: Fit the model
        self.model = model.fit(X_train, y_train)

        # Step 7: Evaluate on the test set
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('saving the model')
        with open("external_surrogate_model.pkl", 'wb') as f:
            pickle.dump(model, f)

        #Step 8: compute spearman correlation
        rho, pvalue = spearmanr(y_test, y_pred)
        print('Spearmans correlation:{}, p-value: {}'.format(rho, pvalue))
        #Step 9: plot the spearman correlation
        plt.scatter(y_test, y_pred)
        plt.xlabel('Hold out set')
        plt.ylabel('Predicted values')
        plt.title('Spearman correlation')
        plt.show()

        return self.model
        
      
    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        default_values = {}
        

        # Get the default value
        default_value = {}
        
        for key, value in self.config_space.items():
            default_values[key] = self.config_space.get_hyperparameter(key).default_value
        
        # Step 2: Find missing keys in the partial configuration
        missing_keys = [key for key in default_values.keys() if key not in theta_new]

        # Step 3: Add missing keys with default values to the configuration
        for key in missing_keys:
            theta_new[key] = default_values[key]

        df = pd.DataFrame([theta_new])
        
        
        y_pred = self.model.predict(df)
       
        #print("This is the prediction ",y_pred)
        return y_pred[0]
        #raise NotImplementedError()


