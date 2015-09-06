from sklearn.metrics import roc_auc_score


def score(y_true, y_pred):
	
	"""
	Returns roc_auc score based on predictions

	Args: 

	y_true: True Predictions
	y_pred: Your Predictions

	Returns:

	score: roc_auc score

	"""

	return roc_auc_score(y_true, y_pred)