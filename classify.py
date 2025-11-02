import numpy as np
import pandas as pd
from scipy.optimize import minimize

class SVM:

    def __init__(self, num_features, C, feature_selection = "entropy"):
        
        """ Initialize the model with the given hyperparameters.

        Args:
            num_features (int): The number of features to choose.
            C (int): The slack parameter for SVM.
            feature_selection (str): The features selection method to use.
        """

        self.num_features = num_features
        self.C = C
        self.feature_selection = feature_selection

        self.ngram_range = None
        self.suffix_range = None
        self.prefix_range = None

        self.df = None
        self.X = None
        self.y = None
        self.w = None
        self.features = None

        self.n = None
        self.d = None

    def select_features(self):
        """Engineers and selects most expressive features. Computes all n-grams 
        of length 1-10 excluding first and last letters. Compute all suffixes 
        and prefixes of length 1-5. Chooses the most expressive features based
        on feature selection criteria. The criteria 'entropy' consistently gave 
        higher k fold cross validation scores.
        """

        # Gets n-grams from a word, does not include first and last letter
        def get_ngrams(word, n):
            return {f"{n}_" + word[i:i+n] 
                    for i in range(1,len(word) - n) if n <= len(word)}

        # Gets n-grams from a series
        def get_all_ngrams(words, n):
            return sorted(list(
                set().union(*[get_ngrams(word, n) for word in words])
            ))

        # Define feature ranges
        self.ngram_range = range(1,11)
        self.suffix_range = range(1,6)
        self.prefix_range = range(1,6)

        # Computes n-grams, suffixes, and prefixes appearing in the dataset
        ngrams = {n: get_all_ngrams(self.df["word"], n) 
                        for n in self.ngram_range}
        suffixes = {s: sorted(list({f"s{s}_" + word[-s:] for word in self.df["word"]})) 
                        for s in self.suffix_range}
        prefixes = {p: sorted(list({f"p{p}_" + word[ :p] for word in self.df["word"]})) 
                        for p in self.prefix_range}

        # One hot encode features in dataframes
        ngram_df = pd.DataFrame(
            [{ngram : 1 if ngram in get_ngrams(word, n) else 0 
            for n in self.ngram_range for ngram in ngrams[n]}
                for word in self.df["word"]])
        suffix_df = pd.DataFrame(
            [{suffix : 1 if suffix == f"s{n}_" + word[-n:] else 0 
            for n in self.suffix_range for suffix in suffixes[n]}
                for word in self.df["word"]])
        prefix_df = pd.DataFrame(
            [{prefix : 1 if prefix == f"p{n}_" + word[:n] else 0 
            for n in self.prefix_range for prefix in prefixes[n]}
                for word in self.df["word"]])

        # Format design matrix and labels
        self.X = pd.concat([ngram_df, suffix_df, prefix_df], axis = 1)
        self.y = self.df["label"].map({"french": 1, "spanish":-1})

        self.n, self.d = self.X.shape
        self.columns = self.X.columns
        

        # If there are less than the max features allowed, use all features
        if self.d <= self.num_features:
            print("Using all features")
            self.features = self.columns


        # Selects features by absolute difference in probability 
        elif self.feature_selection == "probability":

            n_fren = (self.y ==  1).sum()
            n_span = (self.y == -1).sum()

            # Compute posterior distributions with laplace smoothing
            french_post = ((self.X[self.y==1].sum(axis=0) + 1) 
                                    / (n_fren + 2))
            spanish_post= ((self.X[self.y==-1].sum(axis=0) + 1) 
                                    / (n_span + 2))

            # Find most expressive features
            best_feat = np.abs((french_post - spanish_post)
                            / (french_post + spanish_post)
                            ).sort_values(ascending=False)

        # Selects features by absolute minimal entropy in the class labels
        elif self.feature_selection == "entropy":

            # Reformat data
            X_vals = self.X.values
            y_vals = self.y.values.reshape(-1,1)

            # Count of 1's and 0's for each feature
            n1 = X_vals.sum(axis=0)
            n0 = X_vals.shape[0] - n1

            # Masks for where each feature is 1 and 0
            mask1 = X_vals == 1
            mask0 = X_vals == 0

            # Computes probability of labels when features are 1's
            p_fren_x1 = (((y_vals * mask1) == 1).sum(axis = 0) + 1) / (n1 + 2)
            p_span_x1 = (((y_vals * mask1) == -1).sum(axis = 0) + 1) / (n1 + 2)

            # Computes probability of labels when features are 0's
            p_fren_x0 = (((y_vals * mask0) == 1).sum(axis = 0) + 1) / (n0 + 2)
            p_span_x0 = (((y_vals * mask0) == -1).sum(axis = 0) + 1) / (n0 + 2)

            # Compute weighted entropy of the labels given feature values
            S1 = -(
            (p_fren_x1 * np.log(p_fren_x1)) + (p_span_x1 * np.log(p_span_x1))
            )
            S0 = -(
            (p_fren_x0 * np.log(p_fren_x0)) + (p_span_x0 * np.log(p_span_x0))
            )
            S = ((n1 * S1) + (n0 * S0)) / (n1 + n0)

            # Sorts features by minimal entropy 
            best_feat = pd.Series(S, index = self.X.columns).sort_values()

        else:
            print("Invalid feature selection method.")
            return

        # Choose most expressive features
        self.features = best_feat[:self.num_features].index

        self.X = self.X[self.features]
        self.n, self.d = self.X.shape

    def fit(self, df):
        """ Fits model with the training data.

        Args:
            df (DataFrame): Dataframe of words and labels.
        """
        self.df = df

        self.select_features()

        # Format as matrices
        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1,1)
        w_0 = np.zeros(self.d)
        
        # Optimize w_0- gradient descent slow for grid search param optimization
        result = minimize(self._loss_fn, w_0, method='L-BFGS-B', 
                        jac=True, options={"maxiter": 10000})

        if not result.success:
            print("Optimize failed:")
            print(result.message)
            
        # Optimized weights
        self.w = result.x

    def predict(self,df, bagging = False):
        """Predicts the labels in the dataframe.

        Args:
            df (DataFrame): Dataframe containing a "word" column.
            bagging (bool, optional): If the model one of multiple bagged 
            models. Defaults to False.

        Returns:
            Series: A series of the predicted labels.
        """

        # Encode words into feature matrix
        X = self._encode(df["word"])

        # Compute decision values
        decision_values = X @ self.w

        # If not bagging, return decision labels
        if not bagging: 
            
            # Mapping of decision values
            mapping = {1:"french", -1:"spanish", 0:"french"}
            
            return pd.Series(np.sign(decision_values)).map(mapping)
        
        # If bagging, return raw decision values
        else: 

            return pd.Series(decision_values)
        
    # Compute score on a dataframe of words and labels
    def score(self, df, method = 'accuracy'):
        """Scores the model's predictions on a dataframe of words and labels.

        Args:
            df (DataFrame): DataFrame containing "word" and "label" columns.
            method (str, optional): Method to score results. Defaults to 
            'accuracy'.

        Returns:
            float: The accuracy score of the model on the given words and 
            labels.
        """

        if method == 'accuracy':
            return (self.predict(df)==df["label"].reset_index(drop=True)).mean()
        
    def _encode(self, words):

        # Initialize encoded dataframe
        self.X_encode = pd.DataFrame(np.zeros((len(words), self.d), np.uint8), 
                                     columns=self.features)

        # Extract only features that appear in the training data
        word_features = {
            i: self.features.intersection(self._extract_features(word)) 
            for i, word in enumerate(words)
        }

        # One hot encode features
        for index, feature_list in word_features.items():
            self.X_encode.loc[index, feature_list] = 1 

        return self.X_encode.values
            
    # Gets n-grams from a word, does not include first and last letter
    def _extract_features(self, word):

        # Generate n-grams
        n_grams = {
            f"{n}_" + word[i:i+n]
            for n in self.ngram_range
            for i in range(1, len(word) - n) if n <= len(word)
        }

        # Generate suffixes and prefixes
        suffixes = {f"s{s}_" + word[-s:] for s in self.suffix_range}
        prefixes = {f"p{p}_" + word[:p] for p in self.prefix_range}

        # Return as a set
        return n_grams | suffixes | prefixes

    # Get empirical risk with hinge loss
    def _get_risk(self, w):
        w = w.reshape(-1, 1)
        return ((w.T @ w) 
                + (self.C * np.maximum(0, 1 - (self.y * (self.X @ w))).sum())
                )[0,0]
    
    # Get subgradient with hinge loss
    def _get_gradient(self, w):
        w = w.reshape(-1, 1)

        # Get indices contributing to the sub gradient
        ixs =  (self.y * (self.X @ w) < 1).flatten()

        # Compute and return gradient
        return ((2 * w) 
                + (self.C * 
                (-self.y[ixs] * self.X[ixs]).sum(axis = 0)).reshape(-1,1))
    
    # Defines loss function for L-BFGS-B optimization
    def _loss_fn(self, w):
        return self._get_risk(w), self._get_gradient(w)

def classify(train_words, train_labels, test_words):

    # Reformat train and test data
    train = pd.concat([pd.Series(train_words).rename("word"), 
                          pd.Series(train_labels).rename("label")], axis = 1)
    test = pd.DataFrame(test_words).rename(columns = {0:"word"})

    # Set model hyperparameters
    hyperparameters = {
        "C": 0.35,
        "num_features": 250,
        "feature_selection": 'entropy'
    }

    # Fit and predict
    model = SVM(**hyperparameters)
    model.fit(train)
    predictions = list(model.predict(test))

    # return predictions
    return predictions



