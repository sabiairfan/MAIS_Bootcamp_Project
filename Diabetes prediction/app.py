from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SubmitField
import os
import pandas as pd
import numpy as np
from wtforms.validators import ValidationError, InputRequired
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config["SECRET_KEY"] = "sugrh"
Bootstrap(app)

basedir = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(basedir, 'diabetes.csv')
df = pd.read_csv(csv_path)

duplicates = df.duplicated()
num_duplicates = duplicates.sum()

if num_duplicates > 0:
    df = df.drop_duplicates()

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df.loc[df[col] == 0, col] = np.nan

df.fillna(df.mean(), inplace=True)

df1 = df.drop("Outcome", axis =1)

scaler = StandardScaler()
scaler.fit(df1)
scaled_features = scaler.transform(df1)
df_scaled = pd.DataFrame(scaled_features, columns=df1.columns[:])

X = df_scaled
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(50, ), 
    max_iter=1000, 
    random_state=42, 
    alpha=0.001, 
    solver='adam', 
    early_stopping=True,  
    validation_fraction=0.1, 
    n_iter_no_change=10,
    learning_rate_init=0.001,
    )

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],  
    'alpha': [0.0001, 0.001, 0.01],  
    'learning_rate_init': [0.001, 0.01],  
    'max_iter': [1000, 2000],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_model = grid_search.best_estimator_

# Evaluate the accuracy
accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Model accuracy after GridSearchCV: {accuracy:.2f}")

def validate_age(form, field):
    if field.data <=0:
        raise ValidationError('age must be bigger then 0')

class NameForm(FlaskForm):
    pregnancies = IntegerField('Pregnancies', validators=[InputRequired()])
    glucose = IntegerField('Glucose', validators=[InputRequired()])
    bloodpressure = IntegerField('BloodPressure', validators=[InputRequired()])
    skinthickness = IntegerField('SkinThickness', validators=[InputRequired()])
    insulin = IntegerField('Insulin', validators = [InputRequired()])
    bmi = FloatField('BMI', validators=[InputRequired()])
    diabetespedigreefunction = FloatField('DiabetesPedigreeFunction', validators=[InputRequired()])
    age = IntegerField('Age', validators=[InputRequired(), validate_age])
    submit = SubmitField('submit')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = NameForm()

    if form.validate_on_submit():
        data = {
            'Pregnancies': form.pregnancies.data,
            'Glucose': form.glucose.data,
            'BloodPressure': form.bloodpressure.data,
            'SkinThickness': form.skinthickness.data,
            'Insulin': form.insulin.data,
            'BMI': form.bmi.data,
            'DiabetesPedigreeFunction': form.diabetespedigreefunction.data,
            'Age': form.age.data
        }

        input_data = pd.DataFrame([data])
        scaled_input_data = scaler.transform(input_data)
        scaled_input_df = pd.DataFrame(scaled_input_data, columns=df1.columns)
        prediction = best_model.predict(scaled_input_df)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template("result.html", form = form, result = result)
    
    return render_template("home.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)
