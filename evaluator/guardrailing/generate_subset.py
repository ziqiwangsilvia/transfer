import sys

from evaluator.guardrailing.utils import create_balanced_subset, load_data

"""
Make a label balanced dataset subset

"""

data_path = sys.argv[1]
output_filename = sys.argv[2]
n_samples = sys.argv[3]


df = create_balanced_subset(
    dataframe=load_data(data_path=data_path),
    target_col="prompt_label",
    output_filename=output_filename,
    n_samples=int(n_samples),
)
