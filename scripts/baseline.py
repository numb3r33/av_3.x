import numpy as np
"""
Baseline model which always predicts the majority class
"""

def baseline_prediction(train, test):
	majority_class = train.Disbursed.value_counts().argmax()

	return np.array([majority_class * 1.] * test.shape[0])
