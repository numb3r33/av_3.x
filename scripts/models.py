import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC


"""
Baseline model which always predicts the majority class
"""

def baseline_prediction(train, test):
	majority_class = train.Disbursed.value_counts().argmax()

	return np.array([majority_class * 1.] * test.shape[0])


def logistic_regression(features, y):
	"""
	
	Fits a logistic regression model on the data
	Args:
		features: [[x11, x12, ...], [x21, x22, ...]]
		y: [y1, y2, ...]

	Returns:
		model: Logistic Regression model
	
	"""

	est = LogisticRegression(C=1.0, class_weight='auto').fit(features, y)
	return est

def random_forest_classifier(features, y):
	"""

	Fits a random forest classifier with settings for number of estimators
	and class weight

	Args: features, labels
	Returns:
		model: Random Forest Classifier model

	"""

	est = RandomForestClassifier(n_estimators=300, class_weight='auto', oob_score=True).fit(features, y)
	return est


def linear_svc(features, y):
	"""

	Fits a Linear Support Vector classifier with settings for regularization parameter
	and class weight.

	Args: features, labels
	Returns:
		model: Linear Support Vector Classifier model

	"""

	est = LinearSVC(C=.1, class_weight='auto').fit(features, y)
	return est


def gradient_boosting_classifier(features, y):
	"""

	Fits a Gradient Boosting Classifier with settings for number of estimators
	and class weight.

	Args: features, labels
	Returns:
		model: Gradient Boosting Classifier model

	"""

	est = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.9).fit(features, y)
	return est



def predictions(model, features_test):
	"""
	
	Takes in a model and returns predictions

	Args:
		model: can be any model ( Logistic regression model, Random Forest etc.)
		features_test: [[xt11, xt12, ...], [xt21, xt22, ...]]

	Returns:
		preds: predictions on the test set [yt1, yt2, ..]

	"""

	preds = model.predict_proba(features_test)
	return preds
