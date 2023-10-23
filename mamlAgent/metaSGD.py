import random
import numpy as np
import torch
import torch.nn as nn
from learn2learn.algorithms.meta_sgd import MetaSGD


class Agent():
    """
    META-SGD MAML AGENT
    """
    def __init__(self, number_of_algorithms):
        
        self.nA = number_of_algorithms
        
        ####################################

        self.algo_name_list = [str(x) for x in range(number_of_algorithms)]
        self.validation_last_scores = [0.0 for _ in self.algo_name_list]
        self.ds_feat_keys = [
            'target_type', 'task', 'feat_type', 'metric',
            'feat_num', 'target_num', 'label_num', 'train_num',
            'valid_num', 'test_num', 'has_categorical', 'has_missing',
            'is_sparse', 'time_budget'
        ]

        # Initialize the MetaSGD model
        self.meta_model = MetaSGD(nn.Sequential(nn.Linear(len(self.ds_feat_keys), 128), 
                                                nn.ReLU(), 
                                                nn.Linear(128, number_of_algorithms)), lr=0.001)
        self.optimizer = torch.optim.SGD(self.meta_model.parameters(), lr=0.01, momentum=0.9)
        self.loss_criterion = nn.CrossEntropyLoss()

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}

        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """
       
        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.validation_last_scores = [0.0 for i in range(self.nA)]


    """
    META-SGD AGENT TRAIN METHOD
    """
    def meta_train(self, datasets_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent with the validation and test learning curves

        Parameters
        ----------
        datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms

        validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['Erik']
        {'name':'Erik', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'Erik' :

        >>> validation_learning_curves['Erik']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['Erik']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['Erik']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """
        

        """
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        self.datasets_meta_features = datasets_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        """


        META_ITERATIONS = 100  # for example
        INNER_UPDATE_LR = 0.01
        INNER_EPOCHS = 5

        for meta_iteration in range(META_ITERATIONS):
            meta_gradient = None  # reset the accumulated gradient
            for key, ds in validation_learning_curves.items():
                ds_vector = self._get_dataset_vector(datasets_meta_features[key])

                # Clone the model for inner update
                learner = self.meta_model.clone()

                for _ in range(INNER_EPOCHS):
                    # Use the training data to adapt the model
                    # For this example, we're making the assumption that the validation curves can act as training data
                    outputs = learner(torch.Tensor([ds_vector]))
                    loss = self.loss_criterion(outputs, torch.Tensor([int(key)]).long())

                    # Adapt the learner
                    learner.adapt(loss)

                # After adapting, compute loss on validation data (in our case, it's the test curve)
                outputs = learner(torch.Tensor([ds_vector]))
                meta_loss = self.loss_criterion(outputs, torch.Tensor([int(key)]).long())

                # Accumulate the meta-gradient
                meta_gradient = torch.add(meta_gradient, meta_loss) if meta_gradient is not None else meta_loss

            # Update the model based on the accumulated meta-gradient
            self.optimizer.zero_grad()
            meta_gradient.backward()
            self.optimizer.step()
    
    """
    META SGD SUGGEST 
    """

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float)
            The last observation returned by the environment containing:
                (1) A: the explored algorithm,
                (2) C_A: time has been spent for A
                (3) R_validation_C_A: the validation score of A given C_A

        Returns
        ----------
        action : tuple of (int, int, float)
            The suggested action consisting of 3 things:
                (1) A_star: algorithm for revealing the next point on its test learning curve
                            (which will be used to compute the agent's learning curve)
                (2) A:  next algorithm for exploring and revealing the next point
                       on its validation learning curve
                (3) delta_t: time budget will be allocated for exploring the chosen algorithm in (2)

        Examples
        ----------
        >>> action = agent.suggest((9, 151.73, 0.5))
        >>> action
        (9, 9, 80)
        """
        
        """
        #RANDOM CHOICE AGENT
        #=== Uniformly sampling
        next_algo_to_reveal = random.randint(0,self.nA-1)
        delta_t = random.randrange(10, 100, 10)

        if observation==None:
            best_algo_for_test = None
        else:
            A, C_A, R_validation_C_A = observation
            self.validation_last_scores[A] = R_validation_C_A
            best_algo_for_test = np.argmax(self.validation_last_scores)

        action = (best_algo_for_test, next_algo_to_reveal, delta_t)
        return action
        """

        #META-SGD BASED SUGGEST METHOD
        if observation is not None:
            A, _, R_validation_C_A = observation
            self.validation_last_scores[A] = max(self.validation_last_scores[A], R_validation_C_A)

        with torch.no_grad():
            ds_vector = self._get_dataset_vector(self.current_dataset_meta_features)
            scores = self.meta_model(torch.Tensor([ds_vector]))
        
        # Choose the algorithm based on the output scores
        _, predicted = torch.max(scores, 1)
        next_algo_to_reveal = predicted.item()
        
        # For simplicity, let's set a constant delta_t
        delta_t = 10
        action = (next_algo_to_reveal, next_algo_to_reveal, delta_t)
        return action


    def _get_dataset_vector(self, dataset_meta_features):
        values = []
        for k in self.ds_feat_keys:
            v = key_value_mapping(k, dataset_meta_features[k])
            values.append(round(v, 6))
        return values


