##import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

##load dataset
sparcs = pd.read_csv('data_sparcs.csv')
sparcs

##load columns from uploaded dataset
sparcs.columns

##columns of interest
sparcs['Age Group'].describe()

sparcs['APR Risk of Mortality'].describe()

##create new dataframe
sparcs['Age Group'] = pd.to_numeric(sparcs['Age Group'], errors='coerce')
sparcs['age_to_mortality'] = sparcs['Age Group'].apply(lambda x: 'high' if x > 4 else 'low')
sparcs.drop('Age Group', axis=1, inplace=True)
sparcs['age_to_mortality']


##create a new model
x = sparcs.drop(columns=['age_to_mortality'])

y = sparcs['age_to_mortality']

x

y

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.75)

automl = AutoML(results_path='age_to_mortality', mode='explain')
automl

AutoML(mode='explain', results_path='age_to_mortality')

automl.fit(x_train, y_train)

pred = automl.predict(x_test)
pred

automl.report