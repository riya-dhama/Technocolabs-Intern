# importing essential libraries and modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import  train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

"""## Analysing Training Set :"""

# load training set 
train = pd.read_csv("/content/drive/MyDrive/technocolabs training set.csv")
train.shape

"""- It has 218586 rows and 21 attributes """

train.head()

# Information of training set
train.info()

"""- There are 9 float atttributes and 12 object attributes
- As from the data we can see, skip_1, skip_2 and some more coulmns are in  boolean(True or False) but it is counting as object so in pre-processing step we weill change the datatype 
- There are many null values

### Statistical summary
"""

train.describe()

# summary for objects columns
train.describe(include=['O'])

"""### Data-preprocessing and EDA"""

# droping duplicates and null values : 
train.drop_duplicates(inplace=True)
train.dropna(inplace=True)
train.shape

# changing the type of columns 
for colname in ['skip_1','skip_2','skip_3','not_skipped','hist_user_behavior_is_shuffle','premium']:
    train[colname] = train[colname].astype(int, copy=False)

"""### Creating target column"""

train['skip'] = train['not_skipped'].replace({ 0 : 1, 1 : 0 })

train['skip'].value_counts()

train['skip'].value_counts().plot(kind='pie', autopct = "%1.0f%%")

"""### Analysing categorical data"""

col = ['skip_1','skip_2','skip_3',
       'not_skipped','context_switch','no_pause_before_play',
       'short_pause_before_play','long_pause_before_play','hist_user_behavior_is_shuffle',
       'premium','context_type','hist_user_behavior_reason_start',
       'hist_user_behavior_reason_end']

plt.figure(figsize=(20,25))
n = 1
for colname in col:
    plt.subplot(5,3,n)
    train[colname].value_counts().plot(kind='bar')
    plt.xlabel(colname)
    n +=1

# creating copy of the train data 
df = train.copy()
df.shape

# df.date = df.date.apply(pd.to_datetime)
# df.info()

"""### Dropping Irrelevent columns"""

df = df.drop(columns=['skip_1','skip_2','skip_3','not_skipped','date'])
df.shape

df.head()

"""### One-hot Encoding on Train data"""

df1 = df.drop(['session_id', 'track_id_clean'], axis=1)
df1.shape

dummy_train = pd.get_dummies(df1)
dummy_train.shape

dummy_train.head()



"""## Analysing Track Features """

# there are two different files of track features so the aim is to merge both of them then analyse data: 
tf = pd.read_csv('/content/drive/MyDrive/tf_000000000000.csv')
tf.head()

tf2 = pd.read_csv('/content/drive/MyDrive/tf_000000000001.csv')
tf2.head()

# joining two dataframes : 
track = pd.concat([tf, tf2])
track.shape

"""### Inspecting data"""

track.info()

"""- There are 25 float attrbutes, 2 object and 3 integer attributes """

track.duplicated().sum()

track.isna().sum()

"""- There are no duplicate values 
- There are no null values in track_features dataframe

### Statistical summary of track features
"""

track.describe()

# summary for object columns:
track.describe(include='O')

"""### EDA """

# extracting columns other than float ie, int and object for eda :
tf[[c for c in tf.columns if tf[c].dtype != 'float64']].head()

# distribution of release years : 
sns.distplot(track.release_year)
plt.title("Distribution of Release Years");

track['key'].unique()

# we have unique values in keys columns: we have specific Pitch class for these keys(just naming keys acc. to their pitch class)
keys = track.key.value_counts().sort_index()
sns.barplot(
    x=[ "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    y=keys.values/keys.sum()
)
plt.title("Distribution of Track Keys")
plt.xlabel("Key");

track['mode'].value_counts()

track['mode'].value_counts().plot(kind='pie', autopct = "%1.0f%%")

# creating label encoding/one hot encoding of mode column 
track['mode'] = track['mode'].replace({
    'major': 1,
    'minor': 0
})

sns.countplot(track.time_signature)

"""### Analysing numerical columns"""

track.hist(figsize=(20,15));

"""### Correlation matrix  """

plt.figure(figsize=(20,15))
sns.heatmap(tf.corr(), annot=True);

"""### Merging track and Training Data on the basis of track_id (primary key):"""

track.shape

df.shape

df.rename(columns={'track_id_clean': 'track_id'}, inplace=True)

final_train = pd.merge(df, track, on=['track_id'], left_index=True, right_index=False, sort=True)
final_train.shape

final_train.sort_values(axis=0, by=['session_id','session_position'], inplace=True)
final_train.reset_index(drop=True,inplace=True)

final_train.head()

ft = final_train.drop(columns=["session_id","track_id","key", 'time_signature'])
ft = pd.get_dummies(ft, drop_first=True)
ft.shape

ft.info()



"""### Modeling with only Training Data """

dummy_train.head(2)

dummy_train.shape

"""### Train Test Split """

X = dummy_train.drop(columns=["skip"])
y = dummy_train.skip

X_resampled, Y_resampled = SMOTE(sampling_strategy=1.0, random_state=2).fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, Y_resampled,
    test_size=0.2,
    random_state=2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=2
)

"""### Standardsing Data"""

scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)


# Applying LOgistic Regression
log = LogisticRegressionCV(
    cv=3
).fit(
    sX_train,
    y_train
)

print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))

"""###  Applying Random Forest"""

rfc = RandomForestClassifier(
    n_estimators=100
).fit(
    X_train,
    y_train
)

print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))

"""### Feature selection using Boruta """

!pip install boruta
from boruta import BorutaPy

rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)   # initialize the boruta selector
boruta_selector.fit(np.array(sX_train), np.array(y_train))

print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)



selected_rf_features = pd.DataFrame({'Feature':list(X.columns),
                                      'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')

X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))

"""### Creating model with important features (selected by boruta)"""

rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)

print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test))

"""### Applying XG Boost"""

xg = xgb.XGBClassifier()
xg.fit(X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)

print("XGB Train score: %s" % xg.score(X_important_train,y_train))
print("XGB Val score:   %s" % xg.score(X_important_val,y_val))
print("XGB Test score:  %s" % xg.score(X_important_test,y_test))

"""### Testing  Model on the Data (given by technocolabs) """

# loading test data 
ts1 = pd.read_csv('/content/drive/MyDrive/test_data.csv')
ts2 = pd.read_csv('/content/drive/MyDrive/test_data_20.csv')

test_set = pd.concat([ts1,ts2])
test_set.shape

test_set.head(2)

test_set['skip'] =  test_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_test_data = test_set.skip

t1 = test_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean','hist_user_behavior_reason_end_appload'], 
              axis=1)
t1.shape

"""#### Loading Validation data"""

vs1 = pd.read_csv('/content/drive/MyDrive/val_data.csv')
vs2 = pd.read_csv('/content/drive/MyDrive/val_data_20.csv')

val_set = pd.concat([vs1,vs2])
val_set.shape

val_set['skip'] =  val_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_val_data = val_set.skip

v1 = val_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean', 'hist_user_behavior_reason_end_appload'],
             axis=1)
v1.shape

"""### Selecting relevent features from Test and Validation set by using Boruta """

X_val_set = boruta_selector.transform(np.array(v1))
X_test_set = boruta_selector.transform(np.array(t1))

"""### Validation and test score on unseen data"""

print("XGB Val score:   %s" % xg.score(X_val_set, y_val_data))
print("XGB Test score:  %s" % xg.score(X_test_set, y_test_data))

print("RF Val score:   %s" % rf_important.score(X_val_set, y_val_data))
print("RF Test score:  %s" % rf_important.score(X_test_set, y_test_data))



"""### Creating Model with final data (merged data-- ie track + train data)"""

ft.head(2)

X = ft.drop(columns=["skip"])
y = ft.skip

X_resampled, Y_resampled = SMOTE(sampling_strategy=1.0, random_state=2).fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, Y_resampled,
    test_size=0.2,
    random_state=2
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=2
)

"""### Scaling Data"""

scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)


# Applying Logistic Regression
log = LogisticRegressionCV(
    cv=3
).fit(
    sX_train,
    y_train
)

print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))

"""### Applying Random forest Classifier"""

rfc = RandomForestClassifier(
    n_estimators=100
).fit(
    X_train,
    y_train
)

print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))

"""### Feature selction using Boruta"""

rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)   # initialize the boruta selector
boruta_selector.fit(np.array(sX_train), np.array(y_train))

print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)

selected_rf_features = pd.DataFrame({'Feature':list(X.columns),
                                      'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')

"""### Model with important features"""

X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))

"""### Random Forest"""

rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)

print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test))

"""### XG Boost"""

xg = xgb.XGBClassifier()
xg.fit(X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)

print("XGB Train score: %s" % xg.score(X_important_train,y_train))
print("XGB Val score:   %s" % xg.score(X_important_val,y_val))
print("XGB Test score:  %s" % xg.score(X_important_test,y_test))

"""### LGBM """

lgbm = LGBMClassifier( ).fit( X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)

print()
print("LGBM Train score: %s" % lgbm.score(X_important_train,y_train))
print("LGBM Val score:   %s" % lgbm.score(X_important_val,y_val))
print("LGBM Test score:  %s" % lgbm.score(X_important_test,y_test))

y_pred = lgbm.predict(X_important_test)

# draw classification report and confusion matrix  for LGBM MODEL (BASE MODEL)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True)

import pickle
pickle.dump(lgbm, open("lbm.pkl", 'wb'))
pickle.dump(xg, open("xg.pkl", 'wb'))
pickle.dump(rf_important, open("rf.pkl", 'wb'))



"""<h2 align='center'> THANK YOU </h2>"""

