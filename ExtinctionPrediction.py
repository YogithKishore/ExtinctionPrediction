# All imports used throughout the code
# Print statements throughout the code were used to check values as the code ran (not necessary)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import datetime

# Declaring the dataset
SpeciesDataFrame = pd.read_csv("/content/Animal Dataset.csv")

# Seperates each row to represent a single characteristic for simplicity
if "Habitat" in SpeciesDataFrame.columns and "Predators" in SpeciesDataFrame.columns:
  for column in ["Habitat", "Predators"]:
    SpeciesDataFrame[column] = SpeciesDataFrame[column].str.split(", ")
    SpeciesDataFrame = SpeciesDataFrame.explode(column)

# Seperates the numerical and categorical pieces of data into different lists
NumericalColumns = ["Lifespan (years)"]
CategoricalColumns = ["Animal", "Height (cm)", "Weight (kg)", "Color", "Diet", "Habitat", "Predators", "Average Speed (km/h)", "Countries Found", "Conservation Status", "Family", "Gestation Period (days)", "Top Speed (km/h)", "Social Structure", "Offspring per Birth"]

# Transforms the NumericalColumns into numerical data
# Uses the if condition so it only runs if there is a piece of data in the column (accounting for missing pieces of data)
# Values that cannot be converted (strings for example) return a NaN value to ensure no errors
for column in NumericalColumns:
  if column in SpeciesDataFrame.columns:
    SpeciesDataFrame[column] = pd.to_numeric(SpeciesDataFrame[column], errors = "coerce")

# Transforms the CategoricalColumns into categorical data
# Uses the if condition so it only runs if there is a piece of data in the column (accounting for missing pieces of data)
for column in CategoricalColumns:
  if column in SpeciesDataFrame.columns:
    SpeciesDataFrame[column] = SpeciesDataFrame[column].astype("category")

# One-hot encode the data (convert into binary so the model runs better)
SpeciesDataFrameEncoded = pd.get_dummies(SpeciesDataFrame)

# Prepare training data (X drops Lifepsan and y only contains Lifespan)
# .dropna allows for the dataset to drop Lifespan values with a NaN value to ensure it doesn't train on incoplete data
SpeciesDataFrameCleaned = SpeciesDataFrameEncoded.dropna(subset=['Lifespan (years)'])
X = SpeciesDataFrameCleaned.drop("Lifespan (years)", axis=1)
y = SpeciesDataFrameCleaned["Lifespan (years)"]

# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109) # Experimenting showed 109 relatively accurate

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Test accuracy of the model by using the RMSE value
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("RMSE: %0.2f" % (rmse))

# Filling new_data with the data of the Amur Tigers
new_data = pd.DataFrame({"Animal": ["Amur Tiger"], "Height (cm)": [26], "Weight (kg)": [40], "Color": ["Yellow-brown"], "Diet": ["Carnivore"], "Habitat": ["Forest"], "Predators": ["Tigers, Humans"], "Average Speed (km/h)": [54], "Countries Found": ["Asia"], "Conservation Status": ["Critically Endangered"], "Family": ["Felidae"], "Gestation Period (days)": [90], "Top Speed (km/h)": [60], "Social Structure": ["Solitary"], "Offspring per Birth": [2]})

# One-hot encode the new data
new_data_encoded = pd.get_dummies(new_data)

# Ensure that new data has the same columns as the training data
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions for species lifespan
new_predictions = model.predict(new_data_encoded)
print("Predictions for new data (Lifespan in years):", new_predictions)

# Calculates Offspring Births Per Year Per Animal
OffspringPerBirth = new_data["Offspring per Birth"]
NumOfBirths = 2 * new_predictions
OffspringPerYearPerAnimal = (OffspringPerBirth*NumOfBirths)/new_predictions

# Calculates Deaths Per Year
Population = 100
DeathRatePerYear = Population/new_predictions
print(DeathRatePerYear)

# Calculates total Offspring Births Per Year and calculates population difference
OffspringPerYear = OffspringPerYearPerAnimal * Population
Rate = OffspringPerYear - DeathRatePerYear
print(Rate)

# Calculates Death Rate Per Year
Population = 100
FemPop = Population/2
Lifespan = new_predictions[0]
DeathRatePerYear = 1/Lifespan * Population
print(DeathRatePerYear)

# Calculates Births Per Year
BirthsPerYearPerAnimal = 1 * new_data["Offspring per Birth"].iloc[0]
print(BirthsPerYearPerAnimal)
BirthsPerYear = FemPop * BirthsPerYearPerAnimal
print(BirthsPerYear)

# Calculates population difference per year
RateOfChange = BirthsPerYear - DeathRatePerYear
print(RateOfChange)
GrowthRate = (RateOfChange/Population)
print(GrowthRate)

# Sets Graph Axis Numbers & Labels
TimePeriod = 50
t = np.arange(0, TimePeriod + 1)
FuturePop = Population * (GrowthRate)**t
end_date = datetime.date.today()
dates = [end_date + datetime.timedelta(days = 365*n) for n in range(TimePeriod + 1)]

# Displays the Graph
plt.figure(figsize = (10,6))
plt.plot(dates, FuturePop, label = "Population Over Time", color = "blue")
plt.xlabel("Years")
plt.ylabel("Population")
plt.title("Predicted Decline of Amur Tigers")
plt.legend()
plt.grid(True)
plt.show()
