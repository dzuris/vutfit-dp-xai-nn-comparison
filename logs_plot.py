"""Module for generating plots from the logs."""
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_log(filename, target_file, selected_loss_func): # pylint: disable=too-many-branches, too-many-locals
    """
    Parsing the log instance
    
    :param filename: Dataset name
    :param target_file: Targeted dataset logs
    :param selected_loss_func: Targeted loss function logs
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split('--------- START ---------')
    results = defaultdict(lambda: defaultdict(list))  # model -> target -> list of (idx, loss)
    times = defaultdict(lambda: defaultdict(list))    # model -> target -> list of (idx, time)
    model_counters = defaultdict(int)

    for block in blocks:
        if not block.strip():
            continue
        file_match = re.search(r"File:\s*(.+)", block)
        if not file_match or file_match.group(1).strip() != target_file:
            continue

        model_match = re.search(r"Model:\s*(.+)", block)
        loss_func_match = re.search(r"Loss function:\s*'([^']+)'", block)
        loss_values_match = re.findall(r"- ([^:]+): ([\d\.eE+-]+)", block)
        train_time_matches = re.findall(r"- Training time: ([\d\.eE+-]+)", block)

        # Skip log if performance (loss values) are missing or loss function doesn't match
        if not model_match or not loss_func_match or not loss_values_match:
            continue
        loss_func = loss_func_match.group(1).strip()
        if loss_func != selected_loss_func:
            continue

        # Check for negative loss values
        has_negative = False
        parsed_loss_values = []
        for target, value in loss_values_match:
            if target.strip() == "Training time":
                continue  # skip training time as a loss value
            try:
                val = float(value)
                if val < 0:
                    has_negative = True
                parsed_loss_values.append((target, val))
            except ValueError:
                has_negative = True
        if has_negative:
            continue

        model = model_match.group(1).strip()
        idx = model_counters[model]

        # Store only loss values in results
        for target, loss_value in parsed_loss_values:
            results[model][target].append((idx, loss_value))

        # Store only training times in times
        # If there are multiple targets, there should be multiple training times
        for i, (target, _) in enumerate(parsed_loss_values):
            try:
                t = float(train_time_matches[i])
            except (IndexError, ValueError):
                t = None
            times[model][target].append((idx, t))

        model_counters[model] += 1

    return results, times


def plot_results(results, ylabel, title, target_file, filter_none=False):
    """
    Plotting the results
    
    :param results: Results to plot.
    :param ylabel: label of y axis.
    :param title: brief graph description.
    :param target_file: Targeted dataset logs.
    :param filter_none: Only keep points where value is not None.
    """
    plt.figure(figsize=(10, 6))
    colors = {
        'NeuralNetwork': 'tab:blue',
        'GeneticProgramming': 'tab:orange'
    }
    alpha_vals = [1.0, 0.6, 0.3, 0.15]  # For up to 4 targets per model

    for model, targets in results.items():
        for i, (target, vals) in enumerate(targets.items()):
            if filter_none:
                # Only keep points where value is not None
                x = [idx for idx, v in vals if v is not None]
                y = [v for idx, v in vals if v is not None]
            else:
                x = [idx for idx, v in vals]
                y = [v for idx, v in vals]
            if not x or not y:
                continue
            plt.plot(
                x,
                y,
                label=f"{model} - {target}",
                color=colors.get(model, None),
                alpha=alpha_vals[i % len(alpha_vals)],
                linewidth=2,
                marker='o'
            )

    plt.xlabel("Log index")
    plt.ylabel(ylabel)
    plt.title(f"{title} for {target_file}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python logs_plot.py <log_file> <target_file> <loss_function>")
        print("Example: python logs_plot.py logs/training.log ENB2012_data.xlsx mse")
        sys.exit(1)

    log_file_arg = sys.argv[1]
    target_file_arg = sys.argv[2]
    selected_loss_func_arg = sys.argv[3]

    results_global, times_global = parse_log(
        log_file_arg, target_file_arg, selected_loss_func_arg
    )

    if not results_global:
        print(
            f"No results found for file: {target_file_arg} "
            f"with loss function: {selected_loss_func_arg}"
        )
        sys.exit(1)

    # Plot only loss values
    plot_results(
        results_global, selected_loss_func_arg, "Loss values", target_file_arg, filter_none=False
    )
    # Plot only training times (skip None values)
    plot_results(
        times_global, "Training time (s)", "Training time", target_file_arg, filter_none=True
    )
