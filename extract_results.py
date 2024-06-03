########################################
# Script for Parsing ScandEval Results #
########################################

# The MIT License (MIT)
#
# Copyright (c) 2024 - Per Egil Kummervold
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import json
import argparse
import sys
import pandas as pd

# Define the expected datasets and their main metrics
expected_datasets_metrics = {
    "norec": "test_macro_f1",
    "norne-nb": "test_micro_f1",
    "norne-nn": "test_micro_f1",
    "scala-nb": "test_macro_f1",
    "scala-nn": "test_macro_f1",
    "norquad": "test_f1",
    "no-sammendrag": "test_rouge_l",
    "mmlu-no": "test_accuracy",
    "hellaswag-no": "test_accuracy",
    "speed": "Not printed"
}

linguistic_datasets = ["norec", "scala-nn", "no-sammendrag"]
logical_datasets = ["norquad", "mmlu-no", "hellaswag-no"]

def extract_all_results(file_path, output_se, all_metrics):
    results_dict = {}
    existing_models = set()
    unknown_datasets = set()
    unknown_metrics = set()
    conflicts = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                model_name = data.get('model')
                existing_models.add(model_name)
                dataset = data.get('dataset')

                if dataset not in expected_datasets_metrics:
                    unknown_datasets.add(dataset)
                    continue

                if model_name not in results_dict:
                    results_dict[model_name] = {"Dataset": [], model_name: []}

                if dataset and dataset != "speed":
                    metric_key = expected_datasets_metrics[dataset]
                    total_fields = data.get('results', {}).get('total', {})
                    combined_results = []

                    if all_metrics:
                        for k, v in total_fields.items():
                            if k.endswith('_se'):
                                base_key = k[:-3]
                                if base_key in total_fields:
                                    if output_se:
                                        combined_value = f"{round(total_fields[base_key], 2)} ± {round(v, 2)}"
                                    else:
                                        combined_value = f"{round(total_fields[base_key], 2)}"
                                    combined_results.append(combined_value)
                    else:
                        base_value = total_fields.get(metric_key)
                        se_value = total_fields.get(f"{metric_key}_se")
                        if base_value is not None:
                            if output_se and se_value is not None:
                                combined_value = f"{round(base_value, 2)} ± {round(se_value, 2)}"
                            else:
                                combined_value = f"{round(base_value, 2)}"
                            combined_results.append(combined_value)
                        else:
                            unknown_metrics.add(f"{dataset}: {metric_key}")

                    if dataset in results_dict[model_name]["Dataset"]:
                        conflicts.setdefault(model_name, {}).setdefault(dataset, []).append(' / '.join(combined_results))
                    else:
                        results_dict[model_name]["Dataset"].append(dataset)
                        results_dict[model_name][model_name].append(' / '.join(combined_results))

    if conflicts:
        print("Error: Conflicts found for the following models and datasets:")
        for model_name, datasets in conflicts.items():
            for dataset, values in datasets.items():
                print(f"Model: {model_name}, Dataset: {dataset}, Conflicting Values: {values}")
        sys.exit(1)

    if unknown_datasets:
        print(f"Error: Unknown datasets found in the file: {', '.join(unknown_datasets)}")
    if unknown_metrics:
        print(f"Warning: Missing metrics for some datasets: {', '.join(unknown_metrics)}")

    return results_dict, existing_models

def calculate_summary(results_dict, output_se):
    summary = {}
    for model_name, data in results_dict.items():
        linguistic_scores = []
        logical_scores = []

        for dataset, result in zip(data["Dataset"], data[model_name]):
            score_parts = result.split(" / ")
            scores = [float(part.split(" ± ")[0]) for part in score_parts]
            if dataset in linguistic_datasets:
                linguistic_scores.extend(scores)
            elif dataset in logical_datasets:
                logical_scores.extend(scores)

        if linguistic_scores:
            linguistic_avg = round(sum(linguistic_scores) / len(linguistic_scores), 2)
            summary.setdefault(model_name, {})["Linguistic Average"] = linguistic_avg
            if output_se:
                linguistic_se = round(sum([float(part.split(" ± ")[1]) for part in score_parts if " ± " in part]) / len(score_parts), 2)
                summary[model_name]["Linguistic SE"] = linguistic_se

        if logical_scores:
            logical_avg = round(sum(logical_scores) / len(logical_scores), 2)
            summary.setdefault(model_name, {})["Logical Average"] = logical_avg
            if output_se:
                logical_se = round(sum([float(part.split(" ± ")[1]) for part in score_parts if " ± " in part]) / len(score_parts), 2)
                summary[model_name]["Logical SE"] = logical_se

    return summary

def format_markdown_table(results_dict):
    tables = []
    for model_name, data in results_dict.items():
        datasets = data["Dataset"]
        results = data[model_name]
        table = f"| Dataset | {model_name} |\n|:--------|:-------------|\n"
        for dataset, result in zip(datasets, results):
            table += f"| {dataset} | {result} |\n"
        tables.append(table)
    return "\n\n".join(tables)

def display_nicely(results_dict):
    for model_name, data in results_dict.items():
        df = pd.DataFrame(data)
        if df.empty:
            print(f"Results for {model_name} are empty.\n")
        else:
            print(f"Results for {model_name}:\n")
            print(df.to_string(index=False))
            print("\n")

def main():
    parser = argparse.ArgumentParser(description="Extract results from JSONL file for all models.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file")
    parser.add_argument("--markdown", action='store_true', help="Output results in Markdown format")
    parser.add_argument("--output_se", action='store_true', help="Output results with standard error")
    parser.add_argument("--all_metrics", action='store_true', help="Output all metrics instead of the default subset")
    parser.add_argument("--no-summary", action='store_true', help="Do not print summary of scores")
    parser.add_argument("--only-summary", action='store_true', help="Only print summary of scores")
    
    args = parser.parse_args()

    results_dict, existing_models = extract_all_results(args.input_file, args.output_se, args.all_metrics)

    if not args.no_summary or args.only_summary:
        summary = calculate_summary(results_dict, args.output_se)

    if not args.only_summary:
        if args.markdown:
            markdown_output = format_markdown_table(results_dict)
            print(markdown_output)
        else:
            display_nicely(results_dict)

    if not args.no_summary:
        print("\nSummary of Scores:")
        for model_name, scores in summary.items():
            print(f"Model: {model_name}")
            for score_type, value in scores.items():
                print(f"  {score_type}: {value}")

if __name__ == "__main__":
    main()
