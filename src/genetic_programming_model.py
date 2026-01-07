"""GeneticProgrammingModel module.

This module contains implementation of Genetic Programming Model that is subclass of BaseModel.
"""
import os
import random
import functools
import pickle
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from deap import base, creator, tools, gp
from graphviz import Digraph
from base_model import BaseModel
from logging_handler import LoggerHandler
from utils import MODELS_FOLDER, TASK_TYPES, TMP_FOLDER

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
        get_model_summary():
            Summarize the trained model's attributes.
        visualize_model():
            Visualize the model.
        explain_shap():
            Explain model using SHAP method.
        explain_lime() -> list(lime.explanation.Explanation):
            Explain the model using LIME algorithm and returns explanations for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            class_names: list[str],
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
            class_names: list[str]: Unique class names for target column.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str, optional): Name of the file for storing best individuals.
            folder_path (str, optional): Path to folder where GP file should be stored/loaded from.
        """
        super().__init__(
            X=X,
            y=y,
            class_names=class_names,
            config=config,
            logger=logger,
            model_filename=model_filename,
            folder_path=folder_path)
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

    def get_model_summary(self):
        """
        Summarize the genetic programming model for interpretability and save the summary to a file.

        Includes:
        - Decision tree structure (nodes, depth, rules).
        - Complexity metrics (number of rules, branching factor).
        - Summary for each target variable's tree.
        """
        # number of nodes, tree depth
        # number of rules in the tree, average branching factor, redundant or unused nodes/features
        if not self.best_individuals:
            raise ValueError("No trained model exists. Train the model before summarizing.")

        output_file = f"{TMP_FOLDER}/gp_summarization.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Genetic Programming Model Summary\n")
            f.write("=================================\n\n")

            for target_name, tree in self.best_individuals.items():
                f.write(f"Target Variable: {target_name}\n")
                f.write("-----------------------------\n")

                # Number of nodes
                num_nodes = len(tree)
                f.write(f"Number of nodes: {num_nodes}\n")

                # Tree depth
                tree_depth = tree.height
                f.write(f"Tree depth: {tree_depth}\n")

                # Extract rules
                rules = self.extract_rules(tree)
                f.write(f"Number of rules: {len(rules)}\n")
                f.write("Rules:\n")
                for rule in rules:
                    f.write(f"\t{rule}\n")

                # Average branching factor
                branching_factors = [
                    child.arity
                    for child in tree
                    if isinstance(child, gp.Primitive)
                ]

                if branching_factors:
                    avg_branching_factor = sum(branching_factors) / len(branching_factors)
                else:
                    avg_branching_factor = 0
                f.write(f"Average branching factors: {avg_branching_factor:.2f}\n")

                f.write("\n")

            print("Summary saved successfully.")

    def extract_rules(self, tree):
        """
        Extract rules from a genetic programming tree.

        Args:
            tree (gp.PrimitiveTree): The tree to extract rules from.

        Returns:
            list: A list of rules as strings.
        """
        rules = []

        def traverse(index):
            node = tree[index]
            if isinstance(node, gp.Primitive): # Internal node
                rule = f"{node.name}("
                child_rules = [traverse(index + i + 1) for i in range(node.arity)]
                rule += ", ".join(child_rules) + ")"
                return rule

            if isinstance(node, gp.Terminal): # Leaf node
                return str(node.value)

            return ""

        rules.append(traverse(0))
        return rules

    def visualize_model(self):
        """
        Visualize the decision tree(s) trained using genetic programming.

        This function generates a graphical representation of the best individual
        (tree) for each target variable using the `graphviz` library.

        Raises:
            ValueError: If no trained model exists for any target variable.

        Returns:
            dict: A dictionary where keys are target variable names and values are
                  Graphviz Digraph objects representing the trees.
        """
        if not self.best_individuals:
            raise ValueError("No trained model exists. Train the model before visualizing.")

        visualizations = {}

        for target_name, best_individual in self.best_individuals.items():
            # Create a Graphviz Digraph
            dot = Digraph(comment=f"Best Individual for {target_name}")
            dot.attr(dpi='300') # Set high resolution for better visualization

            # Add the root node and recursively add the rest of the tree
            add_nodes_edges(best_individual, dot=dot)

            # Save the visualization for the current target
            visualizations[target_name] = dot

            # Optionally render the graph to a file
            output_file = f"{TMP_FOLDER}/{target_name}_tree"
            dot.render(output_file, format='png', cleanup=True)
            print(f"Visualization for {target_name} saved to {output_file}.png")

        return visualizations

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
        labels = None
        if self.task_type == TASK_TYPES[1]:
            labels = list(range(len(self.class_names)))
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

# Recursive function to add nodes and edges to the graph
def add_nodes_edges(expr, parent=None, tree=None, index=0, dot=None):
    """
    Recursively add nodes and edges to the graph for visualization.

    Args:
        expr: The current node (Primitive or Terminal) to process.
        parent: The parent node of the current node.
        tree: The entire PrimitiveTree structure.
        index: The current index of the node in the tree.

    Returns:
        int: The next index to process in the tree.
    """
    node_id = str(id(expr)) # Unique ID for each node

    # Handle the PrimitiveTree object
    if isinstance(expr, gp.PrimitiveTree): # The entire tree
        root = expr[0] # The root node of the tree

        # Start traversal from the root
        add_nodes_edges(root, parent=None, tree=expr, index=0, dot=dot)
        return 0

    # Handle function nodes (Primitives)
    if isinstance(expr, gp.Primitive): # Function node
        dot.node(node_id, label=expr.name, shape='ellipse')
        if parent:
            print(f"Adding edge from {id(parent)} to {node_id}")
            dot.edge(str(id(parent)), node_id)

        # Recursively process children based on the arity of the current node
        current_index = index + 1
        for _ in range(expr.arity):
            current_index = add_nodes_edges(
                tree[current_index], expr, tree, current_index, dot
                )

        return current_index

    # Handle terminal nodes
    if isinstance(expr, gp.Terminal): # Terminal node
        dot.node(node_id, label=str(expr.value), shape='box')
        if parent:
            dot.edge(str(id(parent)), node_id)
        return index + 1

    # Handle unexpected types
    return index + 1
