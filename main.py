#Install the libraries
!pip install pymongo
!pip install --upgrade pymongo
!python -m pip install "pymongo[srv]
# Import the MongoClient function from the pymongo module
from pymongo import MongoClient

# Create a MongoClient object
client = MongoClient(host=['mongodb+srv://devvashisht111:12345@cluster0.gh2gbeo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'], serverSelectionTimeoutMS=60000)
db = client['salary_db']
#Collection name as salaries
collection = db['salaries']

# Example dataset
data = [
    {"education_level": "Bachelor", "job_role": "Software Engineer", "location": "Mumbai", "industry": "Tech", "salary": 60000},
    {"education_level": "Master", "job_role": "Data Scientist", "location": "Bangalore", "industry": "Tech", "salary": 80000},
    {"education_level": "Bachelor", "job_role": "Marketing Manager", "location": "Delhi", "industry": "Marketing", "salary": 70000},
    {"education_level": "Master", "job_role": "Software Engineer", "location": "Hyderabad", "industry": "Tech", "salary": 75000},
    {"education_level": "PhD", "job_role": "Data Analyst", "location": "Chennai", "industry": "Finance", "salary": 90000},
    {"education_level": "Bachelor", "job_role": "Product Manager", "location": "Pune", "industry": "Tech", "salary": 65000},
    {"education_level": "Master", "job_role": "Software Engineer", "location": "Kolkata", "industry": "Tech", "salary": 78000},
    {"education_level": "Bachelor", "job_role": "Marketing Manager", "location": "Ahmedabad", "industry": "Marketing", "salary": 72000},
    {"education_level": "Master", "job_role": "Data Scientist", "location": "Bangalore", "industry": "Tech", "salary": 82000},
    {"education_level": "Bachelor", "job_role": "Software Engineer", "location": "Mumbai", "industry": "Tech", "salary": 63000},
    {"education_level": "PhD", "job_role": "Data Analyst", "location": "Hyderabad", "industry": "Finance", "salary": 88000},
    {"education_level": "Bachelor", "job_role": "Product Manager", "location": "Delhi", "industry": "Tech", "salary": 66000},
    {"education_level": "Master", "job_role": "Marketing Manager", "location": "Chennai", "industry": "Marketing", "salary": 73000},
    {"education_level": "Bachelor", "job_role": "Software Engineer", "location": "Kolkata", "industry": "Tech", "salary": 76000},
    {"education_level": "Master", "job_role": "Data Scientist", "location": "Pune", "industry": "Tech", "salary": 83000},
    {"education_level": "Bachelor", "job_role": "Marketing Manager", "location": "Ahmedabad", "industry": "Marketing", "salary": 71000},
    {"education_level": "PhD", "job_role": "Software Engineer", "location": "Mumbai", "industry": "Tech", "salary": 64000},
]

Result = collection.insert_many(data)
print(f"Inserted IDs: {Result.inserted_ids}")
#import libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Retrieve data
data = list(collection.find())
df = pd.DataFrame(data)
df = pd.DataFrame(list(collection.find()))
df.head()
# Drop the MongoDB default '_id' column
df = df.drop(columns=['_id'])
# Define features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Identify categorical features
categorical_features = ["education_level", "job_role", "location", "industry"]

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features)
    ])

# Create the pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train the model with the entire dataset
model.fit(X, y)

# Manually define your test set
manual_test_data = {}
for feature in X.columns:
    manual_test_data[feature] = input(f"Enter {feature}: ")

manual_test_df = pd.DataFrame([manual_test_data])

# Predict using the manual test set
predictions = model.predict(manual_test_df)

# Print the predictions for the manual test set
print(f"Predicted salary: {predictions[0]}")
#Import libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Plot bar chart of salary distribution by location
plt.figure(figsize=(12, 6))
sns.barplot(x='location', y='salary', data=df, palette='viridis')#PASSING THE PALETTE VALUES TO HAVE COLOUR SCHEMA
plt.title('Salary Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Salary')
plt.xticks(rotation=45)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
# Plot histogram of salary distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["salary"], bins=20, kde=True, color='skyblue')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()
# Plot bar chart of salary distribution by job role
plt.figure(figsize=(12, 6))
sns.barplot(x='job_role', y='salary', data=df, palette='Set2')
plt.title('Salary Distribution by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Salary')
plt.xticks(rotation=45)
plt.show()
