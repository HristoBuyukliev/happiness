import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes, svm, tree, ensemble
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential

def find_null_percentage(panel):
    null = np.sum(panel.isnull().values)
    return null/float(panel.values.size)

np.random.seed(2466)

# the columns with more than 15 distinct values
categorical = ['V2', 'V144', 'V228', 'V228_2', 'V243_AU', 'V244_AU','V247','V254','V256','V256B','V256C','V256_MAP','V257',
'S024','S025']
useless     = ['Unnamed: 0', 'V2A', 'V248_CS', 'V253_CS',]
discrete    = ['V3','V241','V242','V249','V258','V258A','V261','sacsecval','resemaval','defiance',
'disbelief','relativism','scepticism','equality','choice','voice']
weights     = ['V258','V258A']
target      = ['V10']

# read data
wvs = pd.read_csv('data/wvs.csv',na_values=[-5,-4,-3,-2,-1])

# shuffle
wvs = wvs.sample(frac=1).reset_index(drop=True)

# select only interviews which have happiness answered
wvs = wvs[wvs['V10'].isin([1,2,3,4])]
X = wvs.drop(useless+target+categorical, axis=1).fillna(0)
y = wvs.ix[:,'V10'].values
print 'Original data has ', X.shape[1], ' columns' 

def get_dummy_data(X):
  #find which features are categorical
  count_cols = 0
  cat_cols = []
  for i in X.columns:
      if X.ix[:,i].unique().size <= 11:
          cat_cols.append(i)
          count_cols += X.ix[:,i].unique().size
  print len(cat_cols), ' categorical columns converted into ', count_cols, ' dummy columns'
  # dummify
  X_dumb = pd.get_dummies(X.ix[:,cat_cols+discrete],columns = cat_cols)
  print X_dumb.shape
  # change from pandas DF into numpy arrays
  # X = X.fillna(0).values
  X_dumb = X_dumb.fillna(0).values
  return X_dumb

def get_pca_data(X, num_components=30, train_size=60000):
  pca = decomposition.PCA(n_components=num_components, whiten=True)
  pca.fit(X.values[:train_size,:])
  X_pca = pca.fit_transform(X.values)
  print 'pca complete'
  return X_pca

X_dumb = get_dummy_data(X)
X_pca  = get_pca_data(X, num_components=100)
X = X.values

del wvs, X

# sample_weights = wvs.ix[:,'V258'].values

lb = preprocessing.LabelBinarizer()
y_wide = lb.fit_transform(y)

train_size = 60000

y_train = y[:train_size]
y_test  = y[train_size:]

# logistic regression
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
print cross_val_score(lr,X_pca[:train_size], y_train, scoring='accuracy').mean()
lr.fit(X_pca[:train_size], y_train)
lr.score(X_pca[train_size:], y_test)

# logistic regression with keras - for sanity check
inputs = Input(shape=(30,))
softmax = Dense(4, activation='softmax')(inputs)
model = Model(input=inputs, output=softmax)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_pca[:train_size], y_wide[:train_size], validation_split=0.33)

# extend to mlp
inputs = Input(shape=(100,))
hl1 = Dense(100, activation='relu')(inputs)
dropout = Dropout(0.5)(hl1)
hl2 = Dense(100, activation='relu')(dropout)
softmax = Dense(4, activation='softmax')(hl2)
mlp = Model(input=inputs, output=softmax)
mlp.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
mlp.fit(X_pca[:train_size], y_wide[:train_size], nb_epoch=20)

#Bernoulli Bayes
bnb = naive_bayes.BernoulliNB(alpha=10)
print cross_val_score(bnb, X_dumb[:train_size,:], y_train, scoring='accuracy').mean()
bnb.fit(X_dumb[:train_size], y_train)
print bnb.score(X_dumb[train_size:], y_test)

#Gaussian bayes
gnb = naive_bayes.GaussianNB()
print cross_val_score(gnb, X_pca[:train_size], y[:train_size], scoring='accuracy').mean()
gnb.fit(X_pca[:train_size], y_train)
print gnb.score(X_pca[train_size:], y_test)

#knn
knn = KNeighborsClassifier(n_neighbors = 50, weights='distance')
print cross_val_score(knn, X_pca[:train_size],y[:train_size], scoring='accuracy').mean()
knn.fit(X_pca[:train_size], y_train)
print knn.score(X_pca[train_size:], y_test)

# support vector machines
# all except sigmoid work pretty well, with rbf almost 59% acc.
for kernel in ['linear', 'poly', 'rbf']:
    svc = svm.SVC(kernel=kernel)
    print kernel, cross_val_score(svc, X_pca[:train_size],y[:train_size], scoring='accuracy').mean()
    svc.fit(X_pca[:train_size], y_train)
    print svc.score(X_pca[train_size:], y_test)

#xgboost
xgb = xgboost.XGBClassifier(n_estimators=1000)
print cross_val_score(xgb, X[:train_size], y_train[:train_size], scoring='accuracy').mean()
xgb.fit(X[:train_size], y_train)
print xgb.score(X[train_size:], y_test)

#decision tree
dt = tree.DecisionTreeClassifier()
print cross_val_score(dt, X[:train_size], y_train[:train_size], scoring='accuracy').mean()
dt.fit(X[:train_size], y_train)
print dt.score(X[train_size:], y_test)

#random forest
rf = ensemble.RandomForestClassifier(n_estimators=100,criterion='entropy')
print cross_val_score(rf, X[:train_size], y_train[:train_size], scoring='accuracy').mean()
rf.fit(X[:train_size], y_train)
print rf.score(X[train_size:], y_test)