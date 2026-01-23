"""Tree-based Genetic Programming Model module.

This module contains implementation of a Tree-based Genetic
Programming Model that is a subclass of BaseModel. Moreover
there is _add_nodes_edges function implementation used for generating
tree visualization.
"""
import os
import random
import functools
import pickle
import json
import operator
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
from deap import base, creator, tools, gp, algorithms
from graphviz import Digraph
from src.base_model import BaseModel
from src.logging_handler import LoggerHandler
from src.utils import MODELS_FOLDER, TASK_TYPES, TMP_FOLDER
from src.exceptions import UnsupportedTaskTypeException
from src.gp_primitives import (
    if_then_else,
    if3,
    gt0,
    to_int,
    to_float,
    protected_div,
    protected_sqrt,
    protected_log,
    protected_pow,
    protected_exp,
    protected_reciprocal,
    protected_not
)
from src.gp_types import TInt, TFloat, TBool

class GeneticProgrammingModel(BaseModel):
    """Genetic Programming Model.

    This class implements Genetic Programming Model methods.

    Attributes:
        input_features (list[str]): List of features names.
        pset (gp.PrimitiveSet or gp.PrimitiveSetTypes): Primitive set of functions.
        best_individual (gp.PrimitiveTree): Best trained model.
        toolbox (base.Toolbox): Toolbox for the model.

    Methods:
        _initialize_pset_regression() -> gp.PrimitiveSet:
            Initialize primitive set for regression tasks.
        _initialize_pset_classification() -> gp.PrimitiveSetTyped:
            Initialize primitive set for classification tasks.
        _initialize_toolbox():
            Initialize toolbox for generating GP Tree structure.
        save_model():
            Saves the best model to a file.
        load_model():
            Loads the best model from a file.
        _evaluate(individual, X, y) -> float:
            Calculate fitness function for individual.
        create_and_train_model():
            Creates and train GP model.
        predict(X=None) -> np.ndarray:
            Predicts X data. If X == None then predicts X_test.
        get_model_summary():
            Summarize the trained model's attributes.
        _extract_rules(tree):
            Extract rules from the tree.
        visualize_model():
            Visualize the GP model.
        explain_shap():
            Explain the model using SHAP method.
        explain_lime():
            Explain the model using LIME algorithm.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            y_encoder: LabelEncoder,
            config: dict,
            logger: LoggerHandler,
            model_filename: str = "gp_model.pickle",
            folder_path: str = MODELS_FOLDER):
        """Initialize Tree-based Genetic Programming Model.
        
        The constructor sets model's attribute input_features,
        retype X values into floats and registers fitness functions according to the task type.
        Moreover the initialization of primitive set and toolbox are done here.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            y_encoder (LabelEncoder): Encoder for target column.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str, optional): Name of the file for storing best individual.
            folder_path (str, optional): Path to the folder where
                GP file should be stored/loaded from.
        """
        super().__init__(
            X=X,
            y=y,
            y_encoder=y_encoder,
            config=config,
            logger=logger,
            model_filename=model_filename,
            folder_path=folder_path)

        # Sets input features
        self.input_features = X.columns

        # Retype X data
        self.X_train = self.X_train.astype(float)
        self.X_test = self.X_test.astype(float)

        # Sets fitness functions according to task type
        if self.task_type == TASK_TYPES[0]:     # regression
            if "FitnessMin" not in creator.__dict__:
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

            if "Individual" not in creator.__dict__:
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # pylint: disable=no-member
        elif self.task_type == TASK_TYPES[1]:   # classification
            if "FitnessMin" not in creator.__dict__:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))

            if "Individual" not in creator.__dict__:
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) # pylint: disable=no-member
        else:
            raise UnsupportedTaskTypeException(f"Unsupported task type: {self.task_type}!")

        # Stores best individual
        self.best_individual = None

        # Initialize model's primitive set
        if self.task_type == TASK_TYPES[0]: # regression
            self.pset = self._initialize_pset_regression()
        elif self.task_type == TASK_TYPES[1]:
            self.pset = self._initialize_pset_classification()

        # Initialize model's toolbox
        self.toolbox = self._initialize_toolbox()

    def _initialize_pset_regression(self) -> gp.PrimitiveSet:
        """
        Defines the primitive set for symbolic regression.

        Returns:
            gp.PrimitiveSet: Primitive set.
        """
        pset = gp.PrimitiveSet("MAIN", len(self.input_features))

        # Rename arguments to match columns
        for i, col in enumerate(self.input_features):
            pset.renameArguments(**{f'ARG{i}': col})

        # Arithmetic
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(np.negative, 1)

        # Non-linear
        pset.addPrimitive(np.tanh, 1)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)

        # Polynomial terms
        pset.addPrimitive(np.square, 1)
        pset.addPrimitive(protected_pow, 2)
        pset.addPrimitive(protected_exp, 1)
        pset.addPrimitive(protected_reciprocal, 1)

        # Safe math
        pset.addPrimitive(np.abs, 1)
        pset.addPrimitive(protected_sqrt, 1)
        pset.addPrimitive(protected_log, 1)

        # Constants
        pset.addEphemeralConstant("rand101", functools.partial(random.randint, -1, 1))
        pset.addEphemeralConstant("rand", functools.partial(random.uniform, -5, 5))

        return pset

    def _initialize_pset_classification(self) -> gp.PrimitiveSetTyped:
        """
        Defines the primitive set for classification tasks.

        Returns:
            gp.PrimitiveSetTyped: Primitive set.
        """

        # Output type is TInt (class label)
        input_types = [TFloat] * len(self.input_features)
        pset = gp.PrimitiveSetTyped("MAIN", input_types, TInt)

        # Rename arguments to match columns
        for i, col in enumerate(self.input_features):
            pset.renameArguments(**{f'ARG{i}': col})

        # Boolean operations
        pset.addTerminal(True, TBool)
        pset.addTerminal(False, TBool)
        pset.addPrimitive(operator.lt, [TFloat, TFloat], TBool)
        pset.addPrimitive(operator.gt, [TFloat, TFloat], TBool)
        pset.addPrimitive(operator.le, [TFloat, TFloat], TBool)
        pset.addPrimitive(operator.ge, [TFloat, TFloat], TBool)
        pset.addPrimitive(gt0, [TFloat], TBool)

        pset.addPrimitive(operator.eq, [TInt, TInt], TBool)
        pset.addPrimitive(operator.ne, [TInt, TInt], TBool)

        # Arithmetic float
        pset.addPrimitive(operator.add, [TFloat, TFloat], TFloat)
        pset.addPrimitive(operator.sub, [TFloat, TFloat], TFloat)
        pset.addPrimitive(operator.mul, [TFloat, TFloat], TFloat)
        pset.addPrimitive(operator.neg, [TFloat], TFloat)
        pset.addPrimitive(protected_div, [TFloat, TFloat], TFloat)

        # Non-linearities
        pset.addPrimitive(np.tanh, [TFloat], TFloat)
        pset.addPrimitive(np.abs, [TFloat], TFloat)
        pset.addPrimitive(protected_log, [TFloat], TFloat)
        pset.addPrimitive(protected_sqrt, [TFloat], TFloat)

        # Explicit converters
        pset.addPrimitive(to_int, [TFloat], TInt)
        pset.addPrimitive(to_float, [TInt], TFloat)

        # Logical operators
        pset.addPrimitive(operator.and_, [TBool, TBool], TBool)
        pset.addPrimitive(operator.or_, [TBool, TBool], TBool)
        pset.addPrimitive(protected_not, [TBool], TBool)

        # Conditional if-then-else
        pset.addPrimitive(if_then_else, [TBool, TInt, TInt], TInt)
        pset.addPrimitive(if3, [TBool, TBool, TInt, TInt, TInt], TInt)

        # Class label terminals
        for cls in self.y_encoder.classes_:
            encoded = int(self.y_encoder.transform([cls])[0])
            pset.addTerminal(encoded, TInt)

        # Numeric constants for thresholds
        pset.addEphemeralConstant(
            "rand_float",
            functools.partial(random.uniform, -5.0, 5.0),
            TFloat
        )

        # Add a terminal that accepts any float for compatibility during loading
        pset.addTerminal(0.0, TFloat)

        return pset

    def _initialize_toolbox(self):
        """Setup DEAP toolbox.

        The function creates new toolbox and register basic attributes,
        genetic operators and adds limit for tree height.
        
        Returns:
            Toolbox: Toolbox.
        """

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # pylint: disable=no-member
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # pylint: disable=no-member
        toolbox.register("compile", gp.compile, pset=self.pset)

        # Register genetic operators
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset) # pylint: disable=no-member

        # Add limit tree height to avoid bloat
        max_depth = 20
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_depth))

        return toolbox

    def save_model(self):
        """Saves the trained GP Tree-based model to a file using pickle.

        Raises:
            RuntimeError: If no model is ready for saving.
        """

        # 1. Check if the model exists
        if self.best_individual is None:
            raise RuntimeError(
                "No trained model found. Train the model before saving."
            )

        # 2. Save it
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                "tree_str": str(self.best_individual)
            }, f)

        print(f"Model saved successfully to {self.file_path}")

    def load_model(self):
        """Loads the saved GP Tree-based model from a file.

        Raises:
            FileNotFoundError: If filepath is not leading to any file.
            RuntimeError: If the reconstruction of the model failed.
        """

        # Checks for the file path validity
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f'File for loading tree-based GP model not found: {self.file_path}.'
            )

        # Loads the model
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        model_str = data["tree_str"]

        # Rebuild toolbox
        self.toolbox = self._initialize_toolbox()

        try:
            self.best_individual = gp.PrimitiveTree.from_string(model_str, self.pset)
        except Exception as e:
            print("Error during PrimitiveTree reconstruction:", e)
            print("Model string:", model_str)
            print("Primitives in pset:")
            for primitive in self.pset.primitives[self.pset.ret]:
                print(f"Name: {primitive.name}, Return Type: {primitive.ret}")
            raise RuntimeError(f"Failed to reconstruct GP model: {e}") from e

        print("GP model loaded successfully.")

    def _evaluate(self, individual, X, y, bloat_penalty) -> float:
        """Evaluate individual function.

        The functions calculates mse for regression tasks and accuracy score
        for classification tasks. This is viewed as fitness function
        for evaluating individual on X and y data.
        
        Returns:
            float: Computed mse or accuracy score according to task type.
        """
        # 1. Set task type
        is_regression = self.task_type == TASK_TYPES[0]

        # 2. Compile the GP tree into a callable function
        func = self.toolbox.compile(expr=individual) # pylint: disable=no-member

        # Penalize the bloat
        penalty = len(individual) * bloat_penalty if len(individual) > 15 else 0

        # 3. Make predictions
        try:
            preds = func(*[X[col].values for col in self.input_features])

            # Handle cases where func returns a single scalar instead of an array
            if np.isscalar(preds) or (isinstance(preds, np.ndarray) and preds.ndim == 0):
                preds = np.full(len(y), preds.item() if isinstance(preds, np.ndarray) else preds)

            # Handle invalid results
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                print(
                    "WARNING: NaN/Inf detected in predictions for individual: "
                    f"{str(individual)[:50]}..."
                )
                return (999999.0,) if is_regression else (0.00,)

            if is_regression:
                error = np.mean(np.abs(preds - y.values))
                return (error + penalty,)

            accuracy = accuracy_score(y, np.round(preds))
            return (accuracy - penalty,)

        except ZeroDivisionError as e:
            print(f"ERROR: Division by zero in tree evaluation: {e}")
            return (999999.0,) if is_regression else (0.00, )
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"ERROR: Evaluation failed: {type(e).__name__}: {e}")
            print(f"Individual: {str(individual)[:100]}...")
            return (999999.0,) if is_regression else (0.00, )

    def create_and_train_model(self, training_config: dict):
        """Train the GP Tree-based model.

        The function uses eaSimple algorithm from DEAP library for training the model.
        Best individual is in the end saved to the model's best_individual attribute.

        Args:
            training_config: Model's training configuration.
        """

        # Sets training config
        self.logger.add_log(f"- GP Training configuration: {training_config}")

        # Register evaluation function for this target
        self.toolbox.register(
            "evaluate",
            self._evaluate,
            X=self.X_train,
            y=self.y_train,
            bloat_penalty=training_config['penalty_bloat']
        )

        # Create initial population
        population_size = training_config["population_size"]
        population = self.toolbox.population(n=population_size) # pylint: disable=no-member

        # Elitism settings
        if training_config['elitism']:
            elite_size = max(1, int(population_size * 0.05))
        else:
            elite_size = 1

        hof = tools.HallOfFame(elite_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        # Run evolutionary algorithm
        population, _ = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=training_config["crossover_pb"],
            mutpb=training_config["mutation_pb"],
            ngen=training_config["generations"],
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Store best individual
        best_individual = hof[0]
        self.best_individual = best_individual

        print(f"Best individual for target column {self.target_column}:")
        print(best_individual)

    def predict(self, X=None) -> np.ndarray:
        """
        Predict target values using the trained Tree-based GP model.

        Args:
            X (pd.DataFrame, optional): Input features.
                If None, uses self.X_test.

        Raises:
            ValueError: If best individual model is missing.
            ValueError: If X data and self.X_test data are missing.

        Returns:
            np.ndarray: Predicted values.
        """
        # 1. Ensure model exists
        if self.best_individual is None:
            raise ValueError("Model has not been trained. Train the model before predicting.")

        # 2. Choose input data
        if X is None:
            if self.X_test is None:
                raise ValueError("No X_test provided and no X argument passed to predict().")
            X = self.X_test

        # 3. Compile GP tree into a callable function
        func = self.toolbox.compile(expr=self.best_individual) # pylint: disable=no-member

        # 4. Generate predictions
        pred_values = np.array([func(*row) for _, row in X.iterrows()])

        # 5. Classification: map predictions back to nereast valid class
        if self.task_type == TASK_TYPES[1]: # classification
            classes = self.y_encoder.classes_
            encoded_classes = self.y_encoder.transform(classes)

            pred_values = np.array([
                encoded_classes[np.abs(encoded_classes - p).argmin()]
                for p in pred_values
            ])

        return pred_values

    # --- EXPLAINING FUNCTIONS ---
    def get_model_summary(self):
        """
        Summarizes the GP model for interpretability and saves the summary into a file.

        Includes:
        - Decision tree structure (nodes, depth, rules).
        - Complexity metrics (number of rules, branching factor).
        - Model's loss value.

        Raises:
            ValueError: If no model was trained.
        """
        if self.best_individual is None:
            raise ValueError("No trained model exists. Train the model before summarizing.")

        output_file = f"{TMP_FOLDER}/gp_summarize_{self.target_column}.json"
        gp_summary = {
            "target_column": self.target_column,
            "best_tree": str(self.best_individual),
            "number_of_nodes": len(self.best_individual),
            "depth": self.best_individual.height,
            "rules": _extract_rules(self.best_individual),
            "avg_branching_factor": 0, # placeholder, calculated below
            "loss_func": {
                "name": self.selected_loss,
                "value": self.get_model_loss()
            }
        }

        # Average branching factor
        branching_factors = [
            child.arity
            for child in self.best_individual
            if isinstance(child, gp.Primitive)
        ]

        if branching_factors:
            avg_branching_factor = sum(branching_factors) / len(branching_factors)
        else:
            avg_branching_factor = 0
        gp_summary["avg_branching_factor"] = round(avg_branching_factor, 2)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gp_summary, f, indent=4)

        print(f"Tree-basde GP model summary saved to {output_file}.")

    def visualize_model(self):
        """
        Visualize the decision tree.

        This function generates a graphical representation of the best individual
        (tree) for each target variable using the `graphviz` library.

        Raises:
            ValueError: If no trained model exists for any target variable.
        """
        if self.best_individual is None:
            raise ValueError("No trained model exists. Train the model before visualizing.")

        # Create a Graphviz Digraph
        dot = Digraph(comment=f"Best Individual for {self.target_column}")
        dot.attr(dpi='300') # Set high resolution for better visualization

        # Add the root node and recursively add the rest of the tree
        root = self.best_individual[0]
        _add_nodes_edges(root, tree=self.best_individual, dot=dot)

        # Optionally render the graph to a file
        output_file = f"{TMP_FOLDER}/gp_visualize_{self.target_column}"
        dot.render(output_file, format='png', cleanup=True)
        print(f"Visualization for {self.target_column} saved to {output_file}.png")

    def explain_shap(self): # pylint: disable=signature-differs
        """
        Explain Tree-based GP model using SHAP.

        Raises:
            ValueError: If no model was trained.
        """

        os.makedirs(TMP_FOLDER, exist_ok=True)

        if self.best_individual is None:
            raise ValueError(f"No trained model for target `{self.target_column}`")

        # Compile GP tree for the target
        func = self.toolbox.compile(expr=self.best_individual) # pylint: disable=no-member

        # Define prediction function
        def predict_fn(X):
            """Make predictions using the compiled GP function.
            
            Args:
                X: Input features as DataFrame or numpy array
                
            Returns:
                np.ndarray: Predictions for each row in X
                
            Raises:
                TypeError: If X is neither DataFrame nor numpy array
                RuntimeError: If prediction fails for a row
            """
            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            elif isinstance(X, np.ndarray):
                X_array = X
            else:
                raise TypeError("X must be a pandas DataFrame or numpy array")

            # Apply GP function to each row with error handling
            predictions = []
            for row in X_array:
                try:
                    pred = func(*row)
                    predictions.append(pred)
                except Exception as e:
                    raise RuntimeError(f"Error predicting row {len(predictions)}: {e}") from e

            return np.array(predictions)

        # Calculate shap values
        shap_values = self.calculate_shap_values(
            X=self.X_train,
            predict_fn=predict_fn,
            task_type=self.task_type,
            model_type="gp",
            target_column=self.target_column,
            class_names=self.class_names
        )

        # Summary plot
        shap.summary_plot(
            shap_values,
            features=self.X_train,
            feature_names=self.X_train.columns,
            max_display=10,
            show=False
        )
        figure_file = f"{TMP_FOLDER}/gp_shap_figure_{self.target_column}.png"
        plt.title(f"SHAP Summary Plot — Target: {self.target_column}")
        plt.tight_layout()
        plt.savefig(figure_file)
        plt.close()
        print(f"SHAP figure saved to: {figure_file}")

    def explain_lime(self, instances):
        """
        Explain the GP model using LIME for the given instances.

        Args:
            instances (pd.DataFrame or np.ndarray): Instances to explain.

        Raises:
            ValueError: If no trained model exists.
        """

        # Ensure the model has been trained
        if self.best_individual is None:
            raise ValueError("No trained model exists. Train the model before explaining.")

        # Compile GP tree once
        func = self.toolbox.compile(expr=self.best_individual) # pylint: disable=no-member

        # Define a prediction function for LIME
        def predict_fn(X):
            preds = np.array([func(*row) for row in X])

            if self.task_type == TASK_TYPES[0]: # regression
                return preds.reshape(-1, 1)

            # Classification -> return probability matrix
            class_values = self.y_encoder.transform(self.class_names)
            class_values = np.array(class_values)
            probs = np.zeros((len(preds), len(class_values)))

            # Convert GP numeric output -> one-hot class probabilities
            for i, p in enumerate(preds):
                idx = np.abs(class_values - p).argmin()
                probs[i, idx] = 1.0

            return probs

        # Initiliaze the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=self.class_names if self.task_type == TASK_TYPES[1] else None,
            mode="regression" if self.task_type == TASK_TYPES[0] else "classification",
            random_state=42
        )

        # Ensure the output directory exists
        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Explain each instance and save the explanation to a file
        print(f"LIME Explanation for target column: {self.target_column}")
        for idx in instances:
            data_row = self.X_train.iloc[idx].values

            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                labels=(
                    list(range(len(self.class_names)))
                    if self.task_type == TASK_TYPES[1]
                    else None
                ),
                num_features=min(10, self.X_train.shape[1]) # Limit to 10 features or fewer
            )

            # Save the explanation to a file
            explanation_file = os.path.join(
                TMP_FOLDER, f"gp_lime_{self.target_column}_{idx}.html"
            )
            explanation.save_to_file(explanation_file)
            print(f"Instance: {idx}")
            print(f"- Saved to {explanation_file}")
            print(f"\n- X Data:\n{self.X_train.iloc[idx]}")
            print(f"\n- Y True:\n{self.y_train.iloc[idx]}")

            # Convert explanation to a dictionary
            explanation_dict = {
                "instance": int(idx),
                "target_column": self.target_column,
                "explanation": explanation.as_list(),
                "class_names": self.class_names.tolist() if self.class_names is not None else None,
                "data_row": self.X_train.iloc[idx].to_dict(),
                "target": int(self.y_train.iloc[idx])
            }

            json_file = os.path.join(TMP_FOLDER, f"gp_lime_{self.target_column}_{idx}.json")
            with open(json_file, "w", encoding='utf-8') as f:
                json.dump(explanation_dict, f, indent=4)

            print(f"- Saved JSON explanation to: {json_file}")
            print('-----------------------------------')


def _add_nodes_edges(expr, parent_id=None, tree=None, index=0, dot=None):
    """
    Recursively add nodes and edges to the graph for visualization.

    Args:
        expr: The current node (Primitive or Terminal) for processing.
        parent_id: The node parent's id.
        tree: The entire PrimitiveTree structure.
        index: The current index of the node in the tree.
        dot: Digraph object.

    Returns:
        int: The next index to process in the tree.
    """

    # Unique ID based on tree index
    node_id = f"node_{index}"

    # Forbid PrimitiveTree structures
    if isinstance(expr, gp.PrimitiveTree): # Whole tree sturcture
        raise ValueError(
            "You passed whole tree structure, pass only gp.Primitive or gp.Terminal nodes."
        )

    # Handle function nodes (Primitives)
    if isinstance(expr, gp.Primitive): # Function node
        dot.node(node_id, label=expr.name, shape='ellipse', style='filled', fillcolor='lightblue')
        if parent_id:
            print(f"Adding edge from '{parent_id}' to {node_id}")
            dot.edge(parent_id, node_id)

        # Recursively process children based on the arity of the current node
        current_index = index + 1
        for _ in range(expr.arity):
            current_index = _add_nodes_edges(
                tree[current_index], f"node_{index}", tree, current_index, dot
                )

        return current_index

    # Handle terminal nodes
    if isinstance(expr, gp.Terminal): # Terminal node
        dot.node(node_id, label=str(expr.value), shape='box', style='filled', fillcolor='lightgray')
        if parent_id:
            dot.edge(parent_id, node_id)
        return index + 1

    # Handle unexpected types
    return index + 1


def _extract_rules(tree):
    """
    Extract the rules from a GP tree.

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
