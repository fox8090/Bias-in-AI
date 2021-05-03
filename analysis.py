import warnings
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
import math
import statsmodels.api as sm
import tensorflow as tf
import aif360
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.metrics import confusion_matrix


pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore')

# LOADING
datafile = pd.read_csv("compas-scores-two-years.csv")


# CLEANING
print("\nData cleaning...")
print("Shape of dataset before cleaning:", datafile.shape)
datafile = datafile[(datafile["days_b_screening_arrest"] <= 30)  # Score not calulated within 30 days of arrest
                    # Score not calulated within 30 days of arrest
                    & (datafile["days_b_screening_arrest"] >= -30)
                    & (datafile["is_recid"] != -1)  # No score found
                    # Ordinary traffic offense - no resulting jail time
                    & (datafile["c_charge_degree"] != "O")
                    & (datafile["score_text"] != 'N/A')  # No score found
                    ]

print("Shape of dataset after cleaning:", datafile.shape)
total_entries = datafile.shape[0]


# ANALYSIS
print("\n\n\nAnalysis...")
print("Gender:")
for gender in datafile.sex.unique():
    value = datafile[datafile["sex"] == gender].shape[0]
    print("Number of", gender, "entries: ", value,
          "(" + '%.2g' % (value / total_entries * 100) + "%)")


print("\nRace/Ethnicity:")
for ethnicity in datafile.race.unique():
    value = datafile[datafile["race"] == ethnicity].shape[0]
    print("Number of", ethnicity, "entries: ", value,
          "(" + '%.2g' % (value / total_entries * 100) + "%)")

print("\nGender + Race/Ethnicity:")
for gender in datafile.sex.unique():
    for ethnicity in datafile.race.unique():
        value = datafile[(datafile["sex"] == gender) & (
            datafile["race"] == ethnicity)].shape[0]
        print("Number of", gender, ethnicity, "entries: ", value,
              "(" + '%.2g' % (value / total_entries * 100) + "%)")


print("\nMean of decile scores by race:")
means = datafile.groupby('race')['decile_score'].mean()
print(means)
print("\nVariance of decile score by race:")
var = datafile.groupby('race')['decile_score'].var()
print(var)

print("\n\nAverage age: ", datafile['age'].mean())
print("Variance of age: ", datafile['age'].var())

print("\nRisk Category + Race/Ethnicity:")
for race in datafile.race.unique():
    for tag in datafile.score_text.unique():
        value = datafile[(datafile['race'] == race) & (
            datafile['score_text'] == tag)].shape[0]
        print(tag, 'risk', race, 'entries: ', value,
              "(" + '%.2g' % (value / total_entries * 100) + "%)")

print("\nDecile Score + African American/Caucasian:")
for score in datafile.decile_score.unique():
    value_a = datafile[(datafile['race'] == 'African-American')
                       & (datafile['decile_score'] == score)].shape[0]
    value_c = datafile[(datafile['race'] == 'Caucasian') & (
        datafile['decile_score'] == score)].shape[0]
    print(score, 'risk', 'African American', 'entries: ', value,
          "(" + '%.2g' % (value_a / total_entries * 100) + "%)")
    print(score, 'risk', 'Caucasian', 'entries: ', value,
          "(" + '%.2g' % (value_c / total_entries * 100) + "%)")


# CONVENTIONAL IMPLEMENTATION
print("\n\n\n\n\n\n")

# Drop all columns that will not be used in the model.
datafile = datafile.loc[:, ~datafile.columns.duplicated()]
datafile = datafile.drop(['id', 'dob', 'name', 'first', 'last', 'compas_screening_date', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'r_case_number', 'r_days_from_arrest', 'r_offense_date', 'r_jail_in', 'r_jail_out', 'vr_case_number', 'vr_offense_date', 'screening_date',
                         'v_screening_date', 'in_custody', 'out_custody', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'start', 'end', 'event', 'decile_score.1', 'priors_count.1', 'type_of_assessment', 'c_charge_desc', 'r_charge_desc', 'vr_charge_desc', 'age', 'decile_score', 'violent_recid', 'r_charge_degree', 'vr_charge_degree', 'is_recid', 'is_violent_recid'], axis=1)

# Create dummy variables for categorical data.

categories = ['sex', 'age_cat', 'race', 'score_text', 'c_charge_degree']
for field in categories:
    values = 'field' + '_' + field
    values = pd.get_dummies(datafile[field], prefix=field)
    temp = datafile.join(values)
    datafile = temp

# Drop those dummy variables which are duplicated and combine high and
# medium scores.
datafile['scoreFactor'] = np.where(datafile['score_text_High'] == 1, 1, np.where(
    datafile['score_text_Medium'] == 1, 1, 0))
datafile = datafile.drop(['score_text_High', 'score_text_Medium', 'score_text_Low',
                         'sex_Male', 'age_cat_25 - 45', 'race_Caucasian', 'c_charge_degree_M'], axis=1)

all_fields = datafile.columns.values.tolist()
fields = []
for i in all_fields:
    if i not in categories:
        fields.append(i)

datafile = datafile[fields].astype(int)

# NAIVE LOGISTIC REGRESSION
y = datafile['scoreFactor']
all_fields = datafile.columns.values.tolist()
all_fields.remove('scoreFactor')
X = datafile[all_fields]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

print("Naive logistic regression model...")
x_train = sm.add_constant(X_train)
model = sm.Logit(y_train, x_train)
results = model.fit()

print(results.summary())
predictions = results.predict(pd.concat([X_test, y_test], axis=1)).to_frame()

accuracy = predictions[predictions[0] >= 0.5].shape[0] / predictions.shape[0]
print("Accuracy of model: " + '%.3g' % (accuracy * 100) + "%")

# Values taken from results.summary()
control = math.exp(-1.8289) / (1 + math.exp(-1.8289))
print("African American defendants are " + '%.3g' % (math.exp(0.4337) / (1 - control +
      (control * math.exp(0.4337)))) + " times more likely to receive a higher score.")
print("Female defendants are " + '%.3g' % (math.exp(0.2427) / (1 - control +
      (control * math.exp(0.2427)))) + " times more likely to receive a higher score.")
print("Defendants aged under 25 are " + '%.3g' % (math.exp(1.2263) / (1 - control +
      (control * math.exp(1.2263)))) + " times more likely to receive a higher score.")


# APPLYING SMOTE
print("\n\n\n")
os = SMOTE(random_state=0)
smote_X, smote_y = os.fit_resample(X_train, y_train)
smote_X = pd.DataFrame(data=smote_X, columns=X_train.columns)
smote_y = pd.DataFrame(data=smote_y, columns=['scoreFactor'])

oversampled_df = pd.concat([smote_X, smote_y], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    smote_X, smote_y, test_size=0.3, random_state=1)

x_train = sm.add_constant(X_train)
print("After selcting an unbiased testing set...")
model = sm.Logit(y_train, x_train)
results = model.fit()

print(results.summary())
predictions = results.predict(pd.concat([X_test, y_test], axis=1)).to_frame()

accuracy = predictions[predictions[0] >= 0.5].shape[0] / predictions.shape[0]
print("Accuracy of model: " + '%.3g' % (accuracy * 100) + "%")

# Values taken from results.summary()
control = math.exp(-1.4863) / (1 + math.exp(-1.4863))
print("African American defendants are " + '%.3g' % (math.exp(0.3890) / (1 - control +
      (control * math.exp(0.3890)))) + " times more likely to receive a higher score.")
print("Female defendants are " + '%.3g' % (math.exp(0.0628) / (1 - control +
      (control * math.exp(0.0628)))) + " times more likely to receive a higher score.")
print("Defendants aged under 25 are " + '%.3g' % (math.exp(1.0897) / (1 - control +
      (control * math.exp(1.0897)))) + " times more likely to receive a higher score.")


# Analyse disparate impact
print("\n\n")

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())
predictions = model.predict(X_test)
X_test['predictedScoreFactor'] = predictions
X_test['scoreFactor'] = smote_y

aa_df = X_test[X_test['race_African-American'] == 1]
not_aa_df = X_test[X_test['race_African-American'] == 0]
num_priv = not_aa_df.shape[0]
num_unpriv = aa_df.shape[0]

unprivOutcomes = aa_df[X_test.predictedScoreFactor == 0].shape[0]
print("Ratio of race_African-American favourable outcomes: ",
      unprivOutcomes / num_unpriv)
privOutcomes = not_aa_df[X_test['predictedScoreFactor'] == 0].shape[0]
print(
    "Ratio of not race_African-American favourable outcomes: ",
    privOutcomes /
    num_priv)

disparateImpact = (unprivOutcomes / num_unpriv) / (privOutcomes / num_priv)
print(
    "Disparate impact (race_African-American vs predictedScoreFactor):",
    disparateImpact)

tn, fp, fn, tp = confusion_matrix(
    aa_df['two_year_recid'], aa_df['predictedScoreFactor']).ravel()
print("race_African-American:\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
tn, fp, fn, tp = confusion_matrix(
    not_aa_df['two_year_recid'], not_aa_df['predictedScoreFactor']).ravel()
print("Not race_African-American:\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)


# FAIR MACHINE LEARNING MODEL
print("\n\nFair machine learning model...")
all_fields.append('scoreFactor')
binaryLabelDataset = aif360.datasets.BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=oversampled_df, label_names=[
                                                        'scoreFactor'], protected_attribute_names=['race_African-American'])

dIR = DisparateImpactRemover(repair_level=1.0)
final = dIR.fit_transform(binaryLabelDataset)
transformed = final.convert_to_dataframe()[0]

later = transformed['scoreFactor']
X_final = transformed.drop(['scoreFactor'], axis=1)

y = transformed['scoreFactor']
model = LogisticRegression()
X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
    X_final, y, test_size=0.3, random_state=0)
model.fit(X_final_train, y_final_train)


y_final_pred = model.predict(X_final_test)
X_final_test['predictedScoreFactor'] = y_final_pred
X_final_test['scoreFactor'] = later


accuracy = X_final_test[X_final_test['predictedScoreFactor']
                        == later[0]].shape[0] / X_final_test.shape[0]
print("Accuracy of model: " + '%.3g' % (accuracy * 100) + "%")


aa_df = X_final_test[X_final_test['race_African-American'] == 1]
not_aa_df = X_final_test[X_final_test['race_African-American'] == 0]
num_priv = not_aa_df.shape[0]
num_unpriv = aa_df.shape[0]

unprivOutcomes = aa_df[X_final_test['predictedScoreFactor'] == 0].shape[0]
print("Ratio of race_African-American favourable outcomes: ",
      unprivOutcomes / num_unpriv)
privOutcomes = not_aa_df[X_final_test['predictedScoreFactor'] == 0].shape[0]
print(
    "Ratio of not race_African-American favourable outcomes: ",
    privOutcomes /
    num_priv)

disparateImpact = (unprivOutcomes / num_unpriv) / (privOutcomes / num_priv)
print(
    "Disparate impact (race_African-American vs predictedScoreFactor):",
    disparateImpact)

tn, fp, fn, tp = confusion_matrix(
    aa_df['two_year_recid'], aa_df['predictedScoreFactor']).ravel()
print("race_African-American:\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
tn, fp, fn, tp = confusion_matrix(
    not_aa_df['two_year_recid'], not_aa_df['predictedScoreFactor']).ravel()
print("Not race_African-American:\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
