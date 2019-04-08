# plot prd scores
import os
import json
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("json_files", nargs="*")
parser.add_argument("--output_fig", type=str, default='prd.png')
args = parser.parse_args()

def load_jsons(file_paths):
  scores, labels = [], []
  for json_file in file_paths:
    with open(json_file) as f:
      result = json.load(f)
      scores.append(result["score"])
      labels.append(result["label"])
  return [[s["recall"], s["precision"]] for s in scores], labels

def plot(precision_recall_pairs, labels=None, out_path=None,
         legend_loc='lower left', dpi=300):
  """Plots precision recall curves for distributions.
  Creates the PRD plot for the given data and stores the plot in a given path.
  Args:
    precision_recall_pairs: List of prd_data to plot. Each item in this list is
                            a 2D array of precision and recall values for the
                            same number of ratios.
    labels: Optional list of labels of same length as list_of_prd_data. The
            default value is None.
    out_path: Output path for the resulting plot. If None, the plot will be
              opened via plt.show(). The default value is None.
    legend_loc: Location of the legend. The default value is 'lower left'.
    dpi: Dots per inch (DPI) for the figure. The default value is 150.
  Raises:
    ValueError: If labels is a list of different length than list_of_prd_data.
  """

  if labels is not None and len(labels) != len(precision_recall_pairs):
    raise ValueError(
        'Length of labels %d must be identical to length of '
        'precision_recall_pairs %d.'
        % (len(labels), len(precision_recall_pairs)))

  fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
  plot_handle = fig.add_subplot(111)
  plot_handle.tick_params(axis='both', which='major', labelsize=12)

  for i in range(len(precision_recall_pairs)):
    precision, recall = precision_recall_pairs[i]
    label = labels[i] if labels is not None else None
    plt.plot(recall, precision, label=label, alpha=0.5, linewidth=3)

  if labels is not None:
    plt.legend(loc=legend_loc)

  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.xlabel('Recall', fontsize=12)
  plt.ylabel('Precision', fontsize=12)
  plt.tight_layout()
  plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
  plt.close()

if __name__ == '__main__':
  precision_recall_pairs, labels = load_jsons(args.json_files)
  plot(precision_recall_pairs, labels, args.output_fig)