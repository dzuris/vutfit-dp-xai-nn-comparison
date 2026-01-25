#!/usr/bin/env python3
"""
Comprehensive test script for training models and generating XAI explanations.
Tests multiple datasets with both Neural Network and Genetic Programming models.
Runs all XAI methods: summarize, shap, lime, visualize.
"""

import sys
from datetime import datetime
from pathlib import Path
import argparse
import subprocess
import yaml


# Flag option if models should be retrained using these script
SHOULD_TRAIN_MODEL = True


class TestRunner:
    """Manages comprehensive testing of model training and XAI explanations."""

    # Dataset configurations
    DATASETS = {
        'ENB2012': {
            'path': 'datasets/ENB2012_data.xlsx',
            'type': 'regression',
            'targets': ['Y1', 'Y2'],
            'loss': 'mse',
            'lime_indices': [0, 1, 2]
        },
        'Concrete': {
            'path': 'datasets/concrete+compressive+strength/Concrete_Data.xls',
            'type': 'regression',
            'targets': ['Concrete_compressive_strength_MPa_megapascals'],
            'loss': 'mse',
            'lime_indices': [0, 1, 2]
        },
        'Parkinsons': {
            'path': 'datasets/parkinsons+telemonitoring/parkinsons_updrs.csv',
            'type': 'regression',
            'targets': ['motor_UPDRS', 'total_UPDRS'],
            'loss': 'mse',
            'lime_indices': [0, 1, 2]
        },
        'Iris': {
            'path': 'datasets/iris.csv',
            'type': 'classification',
            'targets': ['variety'],
            'loss': 'accuracy',
            'lime_indices': [0, 1, 2]
        }
    }

    # Model types to test
    MODELS = ['NeuralNetwork', 'GeneticProgramming']

    # XAI methods to test
    XAI_METHODS = ['summarize', 'shap', 'lime', 'visualize']
    # XAI_METHODS = ['summarize', 'lime', 'visualize']

    def __init__(self, output_dir='test_results', quiet=False):
        """Initialize test runner.
        
        Args:
            output_dir: Directory to save test results
            quiet: If True, suppress detailed output
        """
        self.output_dir = Path(output_dir)
        self.quiet = quiet
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'configs').mkdir(exist_ok=True)

        self.results = []

        # Load base configurations (templates)
        self.base_training_config = self._load_base_config('src/training/config_training.yaml')
        self.base_xai_config = self._load_base_config('src/xai/config_xai.yaml')

    def _load_base_config(self, config_path):
        """Load base configuration as template.
        
        Args:
            config_path: Path to config file.
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def log(self, message, level='INFO'):
        """Log message to console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"

        if not self.quiet or level == 'ERROR':
            print(log_message)

        # Append to log file
        log_file = self.output_dir / 'logs' / f'test_{datetime.now().strftime("%Y%m%d")}.log'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def create_training_config(self, dataset_name, model_type):
        """Create temporary training configuration for specific dataset and model.
        
        Args:
            dataset_name: Name of dataset (key in DATASETS dict)
            model_type: 'NeuralNetwork' or 'GeneticProgramming'
        
        Returns:
            Path to created config file
        """
        dataset_info = self.DATASETS[dataset_name]

        # Create a copy of base config
        config = yaml.safe_load(yaml.safe_dump(self.base_training_config))

        # Update dataset info
        config['data']['dataset_path'] = dataset_info['path']
        config['data']['type'] = dataset_info['type']
        config['data']['target_columns'] = dataset_info['targets']

        # Update model selection
        config['selected_model'] = model_type
        config['loss_function'] = dataset_info['loss']

        # Create temporary config file
        config_filename = f'training_{dataset_name}_{model_type}.yaml'
        config_path = self.output_dir / 'configs' / config_filename

        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.log(f"Created training config: {config_filename}")
        return config_path

    def create_xai_config(self, dataset_name, model_type, target, method):
        """Create temporary XAI configuration for specific model and method.
        
        Args:
            dataset_name: Name of dataset
            model_type: 'NeuralNetwork' or 'GeneticProgramming'
            target: Target column name
            method: XAI method ('shap', 'lime', 'visualize', 'summarize')
        
        Returns:
            Path to created config file
        """
        dataset_info = self.DATASETS[dataset_name]

        # Create a copy of base config
        config = yaml.safe_load(yaml.dump(self.base_xai_config))

        # Determine model filename
        model_prefix = 'nn' if model_type == 'NeuralNetwork' else 'gp'
        model_ext = 'keras' if model_type == 'NeuralNetwork' else 'pickle'
        model_filename = f"{model_prefix}_model_{target}.{model_ext}"

        # Update config
        config['data']['dataset_path'] = dataset_info['path']
        config['data']['type'] = dataset_info['type']
        config['data']['target_columns'] = dataset_info['targets']
        config['selected_model'] = model_type
        config['loss_function'] = dataset_info['loss']
        config['xai']['model_filename_to_explain'] = model_filename
        config['xai']['model_target_column_name'] = target
        config['xai']['method'] = method

        # Update LIME indices if needed
        if 'lime' in config['xai']:
            config['xai']['lime']['index_instance_to_explain'] = dataset_info['lime_indices']

        # Create temporary config file
        config_filename = f'xai_{dataset_name}_{model_type}_{target}_{method}.yaml'
        config_path = self.output_dir / 'configs' / config_filename

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        self.log(f"Created XAI config: {config_filename}")
        return config_path

    def run_command(self, command, description):
        """Run shell command and capture output.
        
        Args:
            command: Command to run
            description: Human-readable description
        
        Returns:
            Tuple of (success: bool, output: str)
        """
        self.log(f"Running: {description}")
        self.log(f"Command: {command}", level='DEBUG')

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False
            )

            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully")
                return True, result.stdout

            self.log(f"✗ {description} failed with exit code {result.returncode}", level='ERROR')
            self.log(f"Error output: {result.stderr}", level='ERROR')
            return False, result.stderr

        except subprocess.TimeoutExpired:
            self.log(f"✗ {description} timed out after 10 minutes", level='ERROR')
            return False, "Command timed out"
        except Exception as e: # pylint: disable=broad-exception-caught
            self.log(f"✗ {description} failed with exception: {e}", level='ERROR')
            return False, str(e)

    def train_model(self, dataset_name, model_type):
        """Train a model on specified dataset.
        
        Args:
            dataset_name: Name of dataset
            model_type: 'NeuralNetwork' or 'GeneticProgramming'
        
        Returns:
            bool: Success status
        """
        config_path = self.create_training_config(dataset_name, model_type)

        command = f"python3 -m src.training.main --config {config_path}"
        description = f"Training {model_type} on {dataset_name}"

        success, _ = self.run_command(command, description)

        # Record result
        self.results.append({
            'dataset': dataset_name,
            'model': model_type,
            'operation': 'training',
            'success': success,
            'timestamp': datetime.now().isoformat()
        })

        return success

    def run_xai(self, dataset_name, model_type, target, method):
        """Run XAI explanation on trained model.
        
        Args:
            dataset_name: Name of dataset
            model_type: 'NeuralNetwork' or 'GeneticProgramming'
            target: Target column name
            method: XAI method
        
        Returns:
            bool: Success status
        """
        config_path = self.create_xai_config(dataset_name, model_type, target, method)

        command = f"python3 -m src.xai.main --config {config_path}"
        description = f"Running {method} on {model_type} ({target})"

        success, _ = self.run_command(command, description)

        # Record result
        self.results.append({
            'dataset': dataset_name,
            'model': model_type,
            'target': target,
            'operation': f'xai_{method}',
            'success': success,
            'timestamp': datetime.now().isoformat()
        })

        return success

    def test_dataset(self, dataset_name, models=None):
        """Run complete test suite for a dataset.
        
        Args:
            dataset_name: Name of dataset to test
            models: List of model types to test (None = all)
        
        Returns:
            dict: Summary of results
        """
        self.log("="*70)
        self.log(f"Testing dataset: {dataset_name}")
        self.log("="*70 + "\n")

        dataset_info = self.DATASETS[dataset_name]
        targets = dataset_info['targets']
        models_to_test = models if models else self.MODELS

        summary = {
            'dataset': dataset_name,
            'models_tested': [],
            'total_tests': 0,
            'passed': 0,
            'failed': 0
        }

        for model_type in models_to_test:
            self.log(f"\n--- Testing {model_type} ---\n")

            # Train model
            if SHOULD_TRAIN_MODEL:
                train_success = self.train_model(dataset_name, model_type)
            else:
                train_success = True

            summary['total_tests'] += 1

            if train_success:
                summary['passed'] += 1

                # Run XAI for each target
                for target in targets:
                    self.log(f"\nGenerating explanations for target: {target}")

                    for method in self.XAI_METHODS:
                        xai_success = self.run_xai(dataset_name, model_type, target, method)
                        summary['total_tests'] += 1

                        if xai_success:
                            summary['passed'] += 1
                        else:
                            summary['failed'] += 1
            else:
                summary['failed'] += 1
                self.log(
                    f"Skipping XAI tests for {model_type} due to training failure", level='WARNING'
                )

            summary['models_tested'].append(model_type)

        return summary

    def run_all_tests(self, datasets=None, models=None):
        """Run complete test suite.
        
        Args:
            datasets: List of dataset names to test (None = all)
            models: List of model types to test (None = all)
        
        Returns:
            dict: Complete test results
        """
        datasets_to_test = datasets if datasets else list(self.DATASETS.keys())

        self.log("")
        self.log("="*70)
        self.log("COMPREHENSIVE MODEL TRAINING AND XAI TEST SUITE")
        self.log("="*70)
        self.log(f"Testing datasets: {', '.join(datasets_to_test)}")
        self.log(f"Testing models: {', '.join(models if models else self.MODELS)}")
        self.log(f"XAI methods: {', '.join(self.XAI_METHODS)}")
        self.log("="*70 + "\n")

        all_summaries = []

        for dataset_name in datasets_to_test:
            if dataset_name not in self.DATASETS:
                self.log(f"Unknown dataset: {dataset_name}", level='ERROR')
                continue

            summary = self.test_dataset(dataset_name, models)
            all_summaries.append(summary)

        # Generate final report
        self.generate_report(all_summaries)

        return all_summaries

    def generate_report(self, summaries):
        """Generate and save test report.
        
        Args:
            summaries: List of test summaries
        """
        self.log("\n" + "="*70)
        self.log("TEST SUMMARY REPORT")
        self.log("="*70 + "\n")

        total_passed = sum(s['passed'] for s in summaries)
        total_failed = sum(s['failed'] for s in summaries)
        total_tests = sum(s['total_tests'] for s in summaries)

        for summary in summaries:
            self.log(f"Dataset: {summary['dataset']}")
            self.log(f"  Models tested: {', '.join(summary['models_tested'])}")
            self.log(
                f"  Tests: {summary['total_tests']} total, "
                f"{summary['passed']} passed, {summary['failed']} failed"
            )
            self.log("")

        self.log(f"OVERALL: {total_tests} tests, {total_passed} passed, {total_failed} failed")

        if total_failed == 0:
            self.log("✓ ALL TESTS PASSED!", level='INFO')
        else:
            self.log(f"✗ {total_failed} tests failed", level='WARNING')

        self.log("="*70 + "\n")

        # Save results to YAML
        results_file = self.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump({
                'timestamp': datetime.now().isoformat(),
                'summaries': summaries,
                'detailed_results': self.results,
                'totals': {
                    'tests': total_tests,
                    'passed': total_passed,
                    'failed': total_failed
                }
            }, f, default_flow_style=False, sort_keys=False)

        self.log(f"Detailed results saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive test suite for model training and XAI explanations'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ENB2012', 'Parkinsons', 'Iris', 'all'],
        default='all',
        help='Dataset to test (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['nn', 'gp', 'both'],
        default='both',
        help=(
            'Model type to test: nn (NeuralNetwork), '
            'gp (GeneticProgramming), or both (default: both)'
        )
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_results',
        help='Output directory for test results (default: test_results)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    # Convert model argument
    model_map = {
        'nn': ['NeuralNetwork'],
        'gp': ['GeneticProgramming'],
        'both': None  # None means all models
    }
    models = model_map[args.model]

    # Convert dataset argument
    datasets = None if args.dataset == 'all' else [args.dataset]

    # Run tests
    runner = TestRunner(output_dir=args.output_dir, quiet=args.quiet)
    summaries = runner.run_all_tests(datasets=datasets, models=models)

    # Exit with error code if any tests failed
    total_failed = sum(s['failed'] for s in summaries)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == '__main__':
    main()
