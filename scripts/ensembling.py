import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def create_ensemble_from_submission_files(submission_dir):
	files = os.listdir(submission_dir)
	submissions = {}

	for f in files:
		# don't want to include baseline submission
		if f != 'baseline.csv':
			submission_df = pd.read_csv(submission_dir + f)
			submissions[f] = submission_df.Disbursed

	average_val = np.zeros((37717))
	for f, dis_arr in submissions.iteritems():
		average_val += dis_arr

	return average_val / (len(files) * 1.)


def ranked_averaging(predictions):
	all_ranks = defaultdict(list)

	for i, preds in enumerate(predictions):
		individual_ranks = []

		for e, pred in enumerate(preds):
			individual_ranks.append( (float(pred[1]), e, pred[0]) )

		for rank, item in enumerate( sorted(individual_ranks) ) :
			all_ranks[(item[1], item[2])].append(rank)

	average_ranks = []

	for k in sorted(all_ranks):
		average_ranks.append((sum(all_ranks[k])/len(all_ranks[k]),k))

	ranked_ranks = []

	for rank, k in enumerate(sorted(average_ranks)):
		ranked_ranks.append((k[1][0],k[1][1],rank/(len(average_ranks)-1)))
	return sorted(ranked_ranks)


def stacked_blending(train, y, test):
	X = train
	y = y
	X_submission = test

	skf = list(StratifiedKFold(y, 2))

	clfs = [RandomForestClassifier(n_estimators=300, n_jobs=1, class_weight='auto'),
			GradientBoostingClassifier(learning_rate=0.1, subsample=0.9, max_depth=3, n_estimators=200)]


	print 'Creating train and test sets for blending.'

	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

	for j, clf in enumerate(clfs):
		print j, clf
		dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print "Fold", i
			print train
			X_train = X[train]
			y_train = y[train]
			X_test = X[test]
			y_test = y[test]
			clf.fit(X_train, y_train)
			y_submission = clf.predict_proba(X_test)[:,1]
			dataset_blend_train[test, j] = y_submission
			dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	print
	print "Blending."
	clf = LogisticRegression(class_weight='auto')
	clf.fit(dataset_blend_train, y)
	y_submission = clf.predict_proba(dataset_blend_test)[:,1]

	print "Linear stretch of predictions to [0,1]"
	y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

	return y_submission
