import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Read and Clean Data
df = pd.read_csv("data.csv", skiprows=1)
df.columns = df.columns.str.strip()  # Remove trailing spaces in column names

# Add Region manually based on known data structure
df['Region'] = 0
df.loc[122:, 'Region'] = 1

# Drop columns day, month, year
df.drop(["day", "month", "year"], axis=1, inplace=True, errors='ignore')

# Convert FWI to numeric, coerce errors to NaN
df['FWI'] = pd.to_numeric(df['FWI'], errors='coerce')

# Drop rows with missing values
df = df.dropna().reset_index(drop=True)

# Convert all object columns (except 'Classes') to float
for col in df.columns:
    if df[col].dtype == 'object' and col != "Classes":
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any remaining rows with NaN after conversion
df = df.dropna().reset_index(drop=True)

# Encode Classes Column (0 = fire, 1 = not fire)
df['Classes'] = np.where(df['Classes'].str.contains("not fire", case=False), 1, 0)

# 2. Monthly Fire Analysis in Region 1 (Sidi-Bel-Abbes)
dftemp = df[df["Region"] == 1].copy()

# Add Month_Name column (assuming month was dropped, we can recreate it if needed)
# If month column is needed, ensure it's in the dataset or adjust logic
# Here, we'll assume month is not available, so we'll skip month-based plots unless month is restored
# If you have month data, uncomment and adjust the following:
# dftemp["Month_Name"] = dftemp["month"].apply(lambda x: calendar.month_name[int(x)])
# plt.figure(figsize=(13, 6))
# sns.set_style("whitegrid")
# sns.countplot(x="Month_Name", hue="Classes", data=dftemp, order=calendar.month_name[1:])
# plt.xlabel("Month", fontweight="bold")
# plt.ylabel("Number of Records", fontweight="bold")
# plt.title("Monthly Fire Analysis in Sidi-Bel-Abbes Region", fontsize=20, fontweight="bold")
# plt.xticks(rotation=45)
# plt.legend(title="Class", labels=["Fire", "Not Fire"])
# plt.tight_layout()
# plt.show()

# 3. Separate Fire-Only Monthly Analysis (Region 1)
# fire_only = df[(df["Region"] == 1) & (df["Classes"] == 0)].copy()
# fire_only["Month_Name"] = fire_only["month"].apply(lambda x: calendar.month_name[int(x)])
# plt.figure(figsize=(13, 6))
# sns.set_style("whitegrid")
# sns.countplot(x="Month_Name", data=fire_only, order=calendar.month_name[1:], color='red')
# plt.xlabel("Month", fontweight="bold")
# plt.ylabel("Number of Fires", fontweight="bold")
# plt.title("Monthly Fire-Only Analysis in Sidi-Bel-Abbes Region", fontsize=20, fontweight="bold")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# 4. Feature Selection and Model Training
# Independent and dependent variables
x = df.drop("FWI", axis=1)
y = df["FWI"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to remove highly correlated features
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# Remove correlated features
corr_features = correlation(x_train, 0.85)
x_train.drop(corr_features, axis=1, inplace=True)
x_test.drop(corr_features, axis=1, inplace=True)

# Feature scaling
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Convert scaled train data back to DataFrame for plotting
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)

# # Plot boxplots
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# sns.boxplot(data=x_train)
# plt.title("Before Scaling")
# plt.subplot(1, 2, 2)
# sns.boxplot(data=x_train_scaled_df)
# plt.title("After Scaling")
# plt.tight_layout()
# plt.show()

# 5. Linear Regression
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)
y_pred = regressor.predict(x_test_scaled)

# Metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R2 Score: {r2:.2f}")

# Scatter plot of actual vs predicted
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual FWI", fontweight="bold")
# plt.ylabel("Predicted FWI", fontweight="bold")
# plt.title("Actual vs Predicted FWI", fontsize=16, fontweight="bold")
# plt.tight_layout()
# plt.show()

#lasso regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
lasso=Lasso()
lasso.fit(x_train_scaled,y_train)
y_pred=lasso.predict(x_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R2 Score: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()

from sklearn.linear_model import LassoCV
LassoCV=LassoCV(cv=5)
LassoCV.fit(x_train_scaled,y_train)
y_pred=LassoCV.predict(x_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R2 Score: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()

from sklearn.linear_model import Ridge
Ridge=Ridge()
Ridge.fit(x_train_scaled,y_train)
y_pred=Ridge.predict(x_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()

from sklearn.linear_model import RidgeCV
RidgeCV=RidgeCV(cv=5)
RidgeCV.fit(x_train_scaled,y_train)
y_pred=RidgeCV.predict(x_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error RidgeCv: {mae:.2f}")
# print(f"Mean Squared Error RidgeCv: {mse:.2f}")
# print(f"R2 Score RidgeCv: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()
# print(RidgeCV.get_params())



from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(x_train_scaled,y_train)
y_pred=elastic.predict(x_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error ElasticNet: {mae:.2f}")
# print(f"Mean Squared Error ElasticNet: {mse:.2f}")
# print(f"R2 Score ElasticNet: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()

from sklearn.linear_model import ElasticNetCV
ElasticNetCV=ElasticNetCV(cv=5)
ElasticNetCV.fit(x_train_scaled,y_train)
y_pred=ElasticNetCV.predict(x_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print(f"Mean Absolute Error ElasticNetCV: {mae:.2f}")
# print(f"Mean Squared Error ElasticNetCV: {mse:.2f}")
# print(f"R2 Score ElasticNetCV: {r2:.2f}")
# plt.scatter(y_test,y_pred)
# plt.show()


#pickle the model and prepprocessing model standardscaler
import pickle

pickle.dump(sc,open('scaler.pkl','wb'))
pickle.dump(RidgeCV,open('ridge.pkl','wb'))