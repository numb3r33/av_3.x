import pandas as pd

def create_submission_file(test, predictions, filename):
	
	"""
	Creates a submission file
	"""

	submission_df = pd.DataFrame({'Id': test.index.values, 'Disbursed': predictions})
	submission_df.to_csv('./submission/' + filename, index=False)