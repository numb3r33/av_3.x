import pandas as pd

def create_submission_file(index, predictions, filename):
	
	"""
	Creates a submission file
	"""

	submission_df = pd.DataFrame({'ID': index, 'Disbursed': predictions})
	submission_df.to_csv('./submissions/' + filename, index=False)


def transform_for_ranked(preds):
	ranks = []

	for i, pred in enumerate(preds):
		ranks.append((i, pred))

	return ranks