import pandas as pd
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit


def create_submission_file(index, predictions, filename):
	
	"""
	Creates a submission file
	"""

	submission_df = pd.DataFrame({'ID': index, 'Disbursed': predictions})
	submission_df.to_csv('./submissions/' + filename, index=False)


def transform_for_ranked(preds, index):
	ranks = []

	for i, pred in enumerate(preds):
		ranks.append((index[i], pred))

	return ranks


def cross_val_scores(model, train, y, cv=3):
	sss = StratifiedShuffleSplit(y, n_iter=3, test_size=0.3, random_state=121)
	scores = cross_val_score(model, train, y, cv=sss)

	return scores