"""Models module.

This module contains abstract class for models, with Neural Network and Genetic Programming
models implementations. Also include helper function for obtaining model.
"""
from abc import ABC, abstractmethod
import os
import random
import functools
import pickle
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Neural Network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # pylint: disable=no-name-in-module  # type: ignore

# GP imports
import numpy as np
from deap import base, creator, tools, gp

from logging_handler import LoggerHandler
from utils import calculate_loss, TASK_TYPES, MODELS, TMP_FOLDER
from exceptions import UnsupportedModelException


# Config constants
MODELS_FOLDER = "models"


class BaseModel(ABC): # pylint: disable=too-many-instance-attributes
    """Abstract model class.

    This abstract class provides methods that needs
    to be implemented in subclasses.

    Attributes:
        file_path (str): Path to the model file.
        task_type (str): Type of task the model is dealing with (regression or classification).
        target_columns (list[str]): List of target columns.
        selected_loss (str): Selected loss function (mse, mae for regression;
            accuracy, log_loss for classification)
        target_unique_classes (ndarray): Array containing the
            unique class labels for classification.
        logger (LoggerHandler): Logging handler.
        feature_names (list[str]): List of feature names.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training targets.
        y_test (pd.DataFrame): Testing features.


    Methods:
        save_model():
            Saves model to the file.
        load_model():
            Loads model from the file.
        create_and_train_model():
            Creates and train model on provided data, with config settings.
        predict() -> dict:
            Predict test data.
        get_model_loss() -> float:
            Calculates loss function value on test data.
        explain_shap():
            Explain the model using SHAP values.
        explain_lime() -> list(explanations):
            Explain the model using LIME algorithm and returns explanation for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            config: dict,
            logger: LoggerHandler,
            model_filename: str,
            folder_path: str = MODELS_FOLDER):
        """Model object constructor

        The method creates a file_path to file where model should be stored/loaded from. If the
        path to the file does not exist, the methods creates directories to the file. Moreover
        the function checks if obtained task type is valid and set the task_type attribute. Set
        the target_columns and selected_loss attributes. Checks for number of target columns
        validity and sets target_unique_classes attribute for classification task. Attach
        logger attribute and split data into X_train, X_test, y_train and y_test attributes.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            config (dict): Configuration.
            logger (LoggerHandler): Logger.
            model_filename (str): Filename where model should be saved/loaded from.
            folder_path (str): Path to folder to where model should be saved/loaded from.
        """
        # Set file where to store/load from model
        self.file_path = os.path.join(folder_path, f"{model_filename}")
        os.makedirs(folder_path, exist_ok=True) # Ensure that 'folder_path' folder exists

        # Set task type
        self.task_type = config['data']['type']
        if self.task_type not in TASK_TYPES:
            raise ValueError(
                f"Unsupported task type: {self.task_type}."
                "Task type could be 'regression' or 'classification'.")

        # Obtain other values from config
        self.target_columns = config['data']['target_columns']
        self.selected_loss = config['loss_function']

        # For classification allow only one target column, and obtain number of classes for target
        if self.task_type == TASK_TYPES[1]:
            if y.shape[1] > 1:
                raise ValueError("Only one target column is allowed for classification.")
            # Obtain number of classification classes
            self.target_unique_classes = y.iloc[:, 0].unique()

        # Set logger
        self.logger = logger

        # Set feature names
        self.feature_names = X.columns.tolist()

        # Set datasets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'])

    @abstractmethod
    def save_model(self):
        """Saves model to a file.
        
        This method must be implemented by subclass to define how the model
        should be persisted to disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'save_model' must be implemented in a subclass.")

    @abstractmethod
    def load_model(self):
        """Loads model from a file.
        
        This method must be implemented by subclass to define how the model
        should be loaded from a disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'load_model' must be implemented in a subclass.")

    @abstractmethod
    def create_and_train_model(self):
        """Abstract method for creating and training the model.

        This method must be implemented by subclass to define how to create
        and train the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The method 'create_and_train_model' must be implemented in a subclass.")

    @abstractmethod
    def predict(self) -> dict:
        """Predict data using model.
        
        This method must be implemented by subclass to define how the model
        should make predictions.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'predict' must be implemented in a subclass.")

    def get_model_loss(self) -> float:
        """Method calculates model's loss value.

        Returns:
            float: Calculated loss value.
        """
        predictions = self.predict()
        y_test_dict = self.y_test.to_dict(orient="list")

        print(f"Loss function: {self.selected_loss}")
        return calculate_loss(y_test_dict, predictions, self.selected_loss, self.task_type)

    @abstractmethod
    def explain_shap(self):
        """Explain model using SHAP values.

        This method must be implemented by subclass to define how to explain the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'explain_shap' must be implemented in a subclass.")

    @abstractmethod
    def explain_lime(self, instances):
        "Explain model using LIME method."
        raise NotImplementedError("The method 'explain_lime' must be implemented in a subclass.")


class NeuralNetworkModel(BaseModel):
    """Neural Network Model.

    This class implements Neural Network Model methods.

    Attributes:
        training_config (dict): Configuration for training Neural Network.
        logger (LoggerHandler): Logging Handler.
        model (Sequential): The Neural Network model.

    Methods:
        save_model():
            Saves the Neural Network model to a file.
        load_model():
            Loads the Neural Network model from a file.
        create_and_train_model():
            Creates and train Neural Network model on provided data, with config settings.
        predict():
            Predicts test data.
        explain_shap():
            Explains NN model using SHAP method.
        explain_lime() -> list(lime.explanation.Explanation):
            Explain the model using LIME algorithm and returns explanations for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            config: dict,
            logger: LoggerHandler,
            model_filename: str = "nn_trained_model.keras",
            folder_path: str = MODELS_FOLDER):
        """
        Neural Network model constructor.

        Sets training_config, logger and model attributes.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str): Neural Network model filename.
            folder_path (str): Path to folder where NN model files should be stored/loaded from.
        """
        super().__init__(X, y, config, logger, model_filename, folder_path)
        self.training_config = config['model_training']['nn']
        self.logger.add_log(f"NN Training configuration: {self.training_config}")
        self.model = None

    def save_model(self):
        """Saves model into a file."""
        self.model.save(self.file_path)

    def load_model(self):
        """Loads the model from the file into model attribute.
        
        Raises:
            FileNotFoundError: If filepath is leading to no file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'Cannot find provided file: {self.file_path}.')

        self.model = tf.keras.models.load_model(self.file_path) # pylint: disable=no-member

    def create_and_train_model(self): # pylint: disable=too-many-arguments, too-many-positional-arguments, arguments-differ
        """
        The function creates and train NN model.

        Firstly initializes model's architecture, creates output layer according to task type,
        sets optimizer, compiles the model and trains the model.

        Raises:
            ValueError: If unsupported optimizer is provided in configuration for nn
                optimizer ('adam' or 'sgd' are valid values). Or unsupported type is
                provided ('regression' or 'classification' are valid values).
        """
        self.model = Sequential()

        # Hidden Layers
        for _ in range(self.training_config['hidden_layers']):
            self.model.add(
                Dense(self.training_config['hidden_units'],
                      activation=self.training_config['activation_function']))
            self.model.add(Dropout(self.training_config['dropout_rate']))

        # Output Layer
        if self.task_type == TASK_TYPES[0]:
            # No activation for regression
            number_of_outputs = len(self.target_columns)
            self.model.add(Dense(number_of_outputs, activation='linear'))
            loss_function = "mean_squared_error"
            metrics = ['mae', 'mse']
        elif self.task_type == TASK_TYPES[1]:
            # Classification
            classification_classes_count = len(self.target_unique_classes)
            if classification_classes_count == 1:
                # Binary classification
                self.model.add(Dense(1, activation='sigmoid'))
                loss_function = 'binary_crossentropy'
            else:
                # Multi-class classification
                self.model.add(Dense(classification_classes_count, activation='softmax'))
                loss_function = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Set optimizer
        if self.training_config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=self.training_config['learning_rate'])
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = SGD(learning_rate=self.training_config['learning_rate'])
        else:
            raise ValueError(
                f'Unsupported Neural Network optimizer: {self.training_config['optimizer']}')

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

        # Train the model
        self.model.fit(self.X_train, self.y_train,
                       epochs=self.training_config['epochs'],
                       batch_size=self.training_config['batch_size'],
                       validation_data=(self.X_test, self.y_test),
                       verbose=self.training_config['print_train_logs'])

    def predict(self) -> dict:
        """Predicts test data using the model.

        Returns:
            (dict): Predictions for each target column where column name is the key and
                list of predictions is value.
        """
        # List of predictions
        predictions_nn = self.model.predict(self.X_test)

        # Initialize the dictionary
        result_dict = {target: [] for target in self.target_columns}

        if self.task_type == TASK_TYPES[0]:
            # For regression, return raw values
            for value in predictions_nn:
                for idx, target in enumerate(self.target_columns):
                    result_dict[target].append(value[idx])
        elif self.task_type == TASK_TYPES[1]:
            # Classification
            target_column = self.target_columns[0]
            if self.model.output_shape[-1] == 1:
                # Binary classification: Convert probabilities to class labels (0 or 1)
                predictions_nn = (predictions_nn >= 0.5).astype(int).flatten()
                result_dict[target_column] = predictions_nn.tolist()
            else:
                # Multi-class classification: Get the class index with highest probability
                predictions_nn = np.argmax(predictions_nn, axis=1)
                result_dict[target_column] = predictions_nn.tolist()

        return result_dict

    def explain_shap(self):
        """Explain model using SHAP explainer on train data."""
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(self.model, self.X_train.sample(100, random_state=42))

        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_train)

        # Handle single and multi-output NN shap values
        # For loop iterates through every output, for each output depicts summary
        # graph which feature contributes to the result
        for output_idx, col in enumerate(self.target_columns):
            if shap_values.ndim == 2:   # Regression or single-probability binary
                shap_to_plot = shap_values
            else:
                shap_to_plot = shap_values[:, :, output_idx]

            shap.summary_plot(
                shap_to_plot,
                features=self.X_train,
                feature_names=self.X_train.columns.tolist(),
                show=False
            )
            plt.title(f"SHAP Summary Plot — Target: {col}\n")
            plt.tight_layout()
            plt.show()

    def explain_lime(self, instances):
        """
        Explain the Neural Network model using LIME for the given instances.

        Args:
            instances (pd.DataFrame or np.ndarray): Instances to explain.

        Raises:
            ValueError: If not trained model exists for any target variable.

        Returns:
            list: A list of LIME explanation objects for each instance.
        """
        # Determine mode, class names and labels based on task type
        if self.task_type == TASK_TYPES[1]: # Classification
            mode = "classification"
            class_names = [str(cls) for cls in self.y_train.iloc[:, 0].unique()]
            labels = list(range(len(class_names)))
        elif self.task_type == TASK_TYPES[0]: # Regression
            mode = "regression"
            class_names = None
            labels = None
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Initialize the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=class_names,
            mode=mode,
            random_state=42
        )

        # Ensure the output directory exists
        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Explain each instance and save the explanation to a file
        explanations = []
        for index in range(len(instances)):
            # Generate explanation for the given instance
            data_row = self.X_train.iloc[index].values
            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=self.model.predict,
                labels=labels,
                num_features=self.X_train.shape[1]
            )
            explanations.append(explanation)

            # Save the explanation to a file
            explanation_file = os.path.join(TMP_FOLDER, f"lime_explanation_nn_{index}.html")
            explanation.save_to_file(explanation_file)
            print(f"Explanation for instance {index} saved to {explanation_file}")

        return explanations


class GeneticProgrammingModel(BaseModel):
    """Genetic Programming Model.

    This class implements Genetic Programming Model methods.

    Attributes:
        input_features (list[str]): List of features columns.
        training_config (dict): Configuration for training genetic programming model.
        logger (LoggerHandler): Logging handler.
        toolbox (base.Toolbox): Toolbox for creating GP Tree.
        pset (gp.PrimitiveSet): Primitive set of functions.
        best_individuals (dict): Best trained model for each target variable.

    Methods:
        _initialize_primitive_set():
            Initialize primitive set fitted to input_features.
        _initialize_toolbox():
            Initialize toolbox for generating GP Tree structure.
        save_model():
            Saves the Genetic Programming best individuals to a file.
        load_model():
            Loads the Genetic Programming best individuals from a file.
        _evaluate(individual):
            Calculate MSE as fitness function for individual. 
        create_and_train_model():
            Creates and train Neural Network model.
        predict() -> dict:
            Predicts test data.
        explain_shap():
            Explain model using SHAP method.
        explain_lime() -> list(lime.explanation.Explanation):
            Explain the model using LIME algorithm and returns explanations for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            config: dict,
            logger: LoggerHandler,
            model_filename: str = "gp_model.pickle",
            folder_path: str = MODELS_FOLDER):
        """Initialize Genetic Programming Model.
        
        The constructor sets input_features as X columns, training_config, logger, toolbox, pset
        and best_inidividuals attributes.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str, optional): Name of the file for storing best individuals.
            folder_path (str, optional): Path to folder where GP file should be stored/loaded from.
        """
        super().__init__(X, y, config, logger, model_filename, folder_path)
        self.input_features = X.columns
        self.training_config = config['model_training']['gp']
        self.logger.add_log(f"GP Training configuration: {self.training_config}")
        self.toolbox = base.Toolbox()
        self.pset = self._initialize_primitive_set()
        self.best_individuals = {} # Stores best individuals per target
        self._initialize_toolbox()

    def _initialize_primitive_set(self) -> gp.PrimitiveSet:
        """
        Defines the primitive set for symbolic regression.

        Returns:
            gp.PrimitiveSet: Primitive set.
        """
        pset = gp.PrimitiveSet("MAIN", len(self.input_features))
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(np.negative, 1)
        pset.addEphemeralConstant("rand101", functools.partial(random.randint, -1, 1))

        # Rename arguments to match columns
        for i, col in enumerate(self.input_features):
            pset.renameArguments(**{f'ARG{i}': col})

        return pset

    def _initialize_toolbox(self):
        """Setup DEAP toolbox."""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # pylint: disable=no-member

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual",
                              tools.initIterate,
                              creator.Individual, # pylint: disable=no-member
                              self.toolbox.expr) # pylint: disable=no-member
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) # pylint: disable=no-member
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset) # pylint: disable=no-member
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def save_model(self):
        """Saves GP best individuals converted to string using pickle into a file."""
        with open(self.file_path, 'wb') as f:
            best_individuals_str = {
                target: str(individual) for target, individual in self.best_individuals.items()
                }
            pickle.dump(best_individuals_str, f)

    def load_model(self):
        """Loads the pickle string best individuals.

        Raises:
            FileNotFoundError: If filepath is not leading to any file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f'File for loading genetic programming model not found: {self.file_path}.')

        with open(self.file_path, 'rb') as f:
            # Load the string representation of the best individuals
            best_individuals_str = pickle.load(f)

            # Convert them back to PrimitiveTree objects
            self.best_individuals = {
                target: gp.PrimitiveTree.from_string(individual_str, self.pset)
                for target, individual_str in best_individuals_str.items()
            }

    def _evaluate(self, individual):
        """Evaluate an individual's fitness.

        Args:
            individual: Genetic Programming Tree expression.

        Returns:
            tuple: Predictions in first element, second element is empty
        """
        func = self.toolbox.compile(expr=individual) # pylint: disable=no-member
        predictions = np.array([func(*row) for _, row in self.X_test.iterrows()])

        loss_value = 0
        y = self.y_test.iloc[:, 0].to_numpy()
        if self.task_type == TASK_TYPES[0]:
            # Calculate mean squared error (MSE)
            loss_value = np.mean((predictions - y) ** 2)

        elif self.task_type == TASK_TYPES[1]:
            # Convert predictions to class labels (e.g., 0 or 1)
            unique_classes = self.target_unique_classes
            predicted_labels = np.array(
                [unique_classes[np.abs(unique_classes - p).argmin()] for p in predictions])

            # Calculate Accuracy
            accuracy = np.mean(predicted_labels == y)
            loss_value = 1 - accuracy # Minimize (1 - accuracy) to maximize accuracy

        # Penalize larger trees (penalty term proportional to the tree size)
        tree_size_penalty = len(individual)
        return (loss_value + tree_size_penalty * 0.01,)

    def create_and_train_model(self): # pylint: disable=too-many-locals, arguments-differ
        """Train the GP model for all the target columns."""
        for target_name in self.target_columns:
            # Applying elitism
            if self.training_config['elitism']:
                # Keep top 5% (at least 1 individual)
                elite_size = max(1, int(self.training_config['population_size'] * 0.05))
            else:
                elite_size = 0
            print(f"Elite size: {elite_size}")

            # Settings for training
            self.toolbox.register("evaluate", self._evaluate)
            population = self.toolbox.population(n=self.training_config['population_size']) # pylint: disable=no-member

            print(f'Training GP for {target_name}')

            best_overall_individual = None

            # Run the evolutionary algorithm
            for gen in range(self.training_config['generations']):

                # Select elites (best individuals from the current population)
                elites = list(map(self.toolbox.clone, tools.selBest(population, elite_size))) # pylint: disable=no-member

                # Select the rest using tournament selection
                offspring = self.toolbox.select(population, len(population) - elite_size) # pylint: disable=no-member
                offspring = list(map(self.toolbox.clone, offspring)) # pylint: disable=no-member

                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.training_config['crossover_pb']:
                        self.toolbox.mate(child1, child2) # pylint: disable=no-member
                        del child1.fitness.values, child2.fitness.values

                # Apply mutation
                for mutant in offspring:
                    if random.random() < self.training_config['mutation_pb']:
                        self.toolbox.mutate(mutant, expr=self.toolbox.expr) # pylint: disable=no-member
                        del mutant.fitness.values

                # Evaluate new individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind) # pylint: disable=no-member
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Combine elites and new offsprings
                population[:] = elites + offspring

                # Track overall best individual
                best_individual = max(population, key=lambda ind: ind.fitness.values)
                if gen == 0 or best_individual.fitness.values[0] < best_overall_fitness:
                    best_overall_fitness = best_individual.fitness.values[0]
                    best_overall_individual = best_individual

                # Print the best fitness from generation each generation
                print(f"Generation {gen} "
                    f"- Best Fitness: {best_individual.fitness.values[0]} "
                    f"- Overall Best Fitness: {best_overall_fitness}")

            # Store the best individual
            self.best_individuals[target_name] = best_overall_individual
            print(f"Best individual for {target_name}")
            print(best_overall_individual)
            print()

    def predict(self) -> dict:
        """
        Make predictions using the best model for each target variable.

        Raises:
            ValueError: If best individuals dictionary does not contain best model
                for any target variable.

        Returns:
            (dict): Dictionary of predictions where target column names are keys and list of
                predictions are values for each target column.
        """
        predictions = {}
        for target_name in self.target_columns:

            # Check if is there a tree for target column
            if target_name not in self.best_individuals:
                raise ValueError(f"Model for target '{target_name}' has not been trained.")

            # Obtain tree for predicting target column
            best_ind = self.best_individuals[target_name]
            func = self.toolbox.compile(expr=best_ind) # pylint: disable=no-member
            pred_values = np.array([func(*x) for _, x in self.X_test.iterrows()])

            if self.task_type == TASK_TYPES[1]:
                # Classification
                # Convert to nearest valid class
                unique_classes = self.target_unique_classes # Ensure we use correct class labels
                pred_values = np.array(
                    [unique_classes[np.abs(unique_classes - p).argmin()] for p in pred_values])

            predictions[target_name] = pred_values
        return predictions

    def explain_shap(self): # pylint: disable=signature-differs
        """Explain Genetic Programming models using SHAP for each target column."""
        max_display=10
        background_size = 100

        for target_name in self.target_columns:
            print(f"Explaining GP model for target: {target_name}")

            if target_name not in self.best_individuals:
                raise ValueError(f"No trained model for target `{target_name}`")

            # Compile GP tree for the target
            best_ind = self.best_individuals[target_name]
            func = None
            if hasattr(self.toolbox, "compile"):
                func = self.toolbox.compile(expr=best_ind)
            else:
                raise AttributeError("Toolbox does not have a 'compile' method")

            # Define prediction function
            def predict_fn(X):
                if isinstance(self.X_train, pd.DataFrame):
                    return np.array([func(*x) for x in X]) # pylint: disable=W0640
                raise ValueError("X_train must be a pandas DataFrame")

            # Background data (sampled subset of training data)
            background = self.X_train.sample(n=min(background_size, len(self.X_train)),
                                             random_state=42).values

            # Initialize SHAP Kernel Explainer
            explainer = shap.KernelExplainer(predict_fn, background)

            # Compute SHAP values
            shap_values = explainer.shap_values(self.X_train.values)

            # Summary plot
            shap.summary_plot(shap_values,
                              self.X_train,
                              feature_names=self.X_train.columns,
                              max_display=max_display)

    def explain_lime(self, instances):
        """
        Explain the Genetic Programming model using LIME for the given instances.

        Args:
            instances (pd.DataFrame or np.ndarray): Instances to explain.

        Raises:
            ValueError: If not trained model exists for any target variable.

        Returns:
            list: A list of LIME explanation objects for each instance.
        """

        # Ensure the model has been trained
        if not self.best_individuals:
            raise ValueError("No trained model exists. Train the model before explaining.")

        # Define a prediction function for LIME
        def predict_fn(X):
            predictions = []
            for target_name in self.target_columns:
                if target_name not in self.best_individuals:
                    raise ValueError(f"No trained model for target `{target_name}`")
                best_ind = self.best_individuals[target_name]
                if hasattr(self.toolbox, "compile"):
                    func = self.toolbox.compile(expr=best_ind) # Compile the GP tree
                else:
                    raise AttributeError("Toolbox does not have a 'compile' method")
                pred_values = np.array([func(*x) for x in X])

                # For classification, ensure probabilities are returned
                if self.task_type == TASK_TYPES[1]: # Classification
                    unique_classes = self.target_unique_classes
                    probabilities = np.zeros((len(pred_values), len(unique_classes)))
                    for i, p in enumerate(pred_values):
                        # Assign probabilities based on the closest class
                        closest_class_idx = np.abs(unique_classes - p).argmin()
                        probabilities[i, closest_class_idx] = 1.0
                    predictions.append(probabilities)
                else:
                    predictions.append(pred_values)

            if self.task_type == TASK_TYPES[0]: # Regression
                return np.column_stack(predictions)

            # Classification
            return predictions[0]

        # Initiliaze the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            mode="regression" if self.task_type == TASK_TYPES[0] else "classification",
            random_state=42
        )

        # Ensure the output directory exists
        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Explain each instance and save the explanation to a file
        explanations = []
        class_names = None
        labels = None
        if self.task_type == TASK_TYPES[1]:
            class_names = [str(cls) for cls in self.y_train.iloc[:, 0].unique()]
            labels = list(range(len(class_names)))
        for i, instance in enumerate(instances):
            data_row = self.X_train.iloc[instance].values
            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                labels=labels,
                num_features=min(10, len(self.input_features)) # Limit to 10 features or fewer
            )
            explanations.append(explanation)

            # Save the explanation to a file
            explanation_file = os.path.join(TMP_FOLDER, f"lime_explanation_gp_{i}.html")
            explanation.save_to_file(explanation_file)
            print(f"Explanation for instance {i} saved to {explanation_file}")

        # Return the list of explanation objects
        return explanations


def get_model(selected_model: str, X: pd.DataFrame, y: pd.DataFrame,
              config: dict, logger: LoggerHandler) -> BaseModel:
    """Obtain selected model.

    The function initializes Model object with all necessary arguments.

    Args:
        selected_model (str): Model selection ('NeuralNetwork' or 'GeneticProgramming').
        X (pd.DataFrame): Dataset Features.
        y (pd.DataFrame): Dataset Targets.
        config (dict): Configuration settings.
        logger (LoggerHandler): Handler for logging messages.

    Raises:
        UnsupportedModelException: If unsupported model selection si provided.

    Returns:
        BaseModel: Model object.
    """
    models = {
        MODELS[0]: NeuralNetworkModel,
        MODELS[1]: GeneticProgrammingModel
    }

    if selected_model not in models:
        raise UnsupportedModelException(
            f"Unsupported model selection: {selected_model}."
            f"Possible values: '{MODELS[0]}' or '{MODELS[1]}'.")

    return models[selected_model](X, y, config, logger)
