from sklearn.preprocessing import LabelEncoder

def process(train, test, cols):
	"""
	Encodes columns with labels to ints
	e.g. Male/Female gets converted to 1/0
	"""

	for col in cols:
		if col != 'DOB' and col != 'Lead_Creation_Date':
			lbl = LabelEncoder()
			
			train_unique = list(train[col].unique())
			test_unique = list(test[col].unique())

			train_unique.extend(test_unique)
			lbl.fit(train_unique)

			train[col] = lbl.transform(train[col])
			test[col] = lbl.transform(test[col])

	return [train, test]


def not_null_cols(train):
	cols = train.columns
	not_null = []

	for col in cols[:-2]:
		if train[col].isnull().any() == False:
			not_null.append(col)


	return not_null

def cols_with_obj_type(train, cols):
	obj_cols = []

	for col in cols:
		if train[col].dtype == 'object':
			obj_cols.append(col)

	return obj_cols