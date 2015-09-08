import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

def feature_selection(train, y):

	sss = StratifiedShuffleSplit(y, n_iter=1, test_size=.3, random_state=42)
	train_idx, test_idx = next(iter(sss))

	xtrain = train.iloc[train_idx].values
	ytrain = y.iloc[train_idx].values

	xtest = train.iloc[test_idx].values
	ytest = y.iloc[test_idx].values

	clf_et = ExtraTreesClassifier().fit(xtrain, ytrain)

	et_preds = clf_et.predict(xtest)

	print 'initial f1 score based on extra trees classifier: ', f1_score(ytest, et_preds)

	feat_imp = clf_et.feature_importances_
	sorted_fi = feat_imp[np.argsort(feat_imp)[::-1]] #descending sort

	print 'feature importance: ', feat_imp 
	print 'sorted feature importances: ', sorted_fi

	clf_gb = GradientBoostingClassifier()
	feats_tot = xtrain.shape[1]

	f1_best = 0
	print "output format:"
	print "no of features, f1-score, roc-score of class-predictions, roc-score of probabilities"

	for feats in range(1,feats_tot+1):
		threshold_idx = min(len(sorted_fi),feats)
		threshold = sorted_fi[threshold_idx]
		select = (feat_imp>threshold)
		clf_gb.fit(xtrain[:,select],ytrain)
		tmp_preds = clf_gb.predict(xtest[:,select])
		tmp_probs = clf_gb.predict_proba(xtest[:,select])[:,1]
		f1 = f1_score(ytest,tmp_preds)
		roc_pred = roc_auc_score(ytest,tmp_preds)
		roc_prob = roc_auc_score(ytest,tmp_probs)
		if f1 > f1_best:
			f1_best = f1
			np.save('./features/clf_sel.npy',select)
		print feats,f1,roc_pred,roc_prob
		if feats >= 16:
			break

	print "f1_best:", f1_best
