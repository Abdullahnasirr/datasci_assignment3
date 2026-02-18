import pandas as pd

#this Loads the dataset
df = pd.read_csv("crime1.csv")

#this will Determine the average, median, standard deviation, maximum and minimum values
average_violent_crime = df["ViolentCrimesPerPop"].mean()

median_violent_crime = df["ViolentCrimesPerPop"].median()

standard_deviation_violent_crime = df["ViolentCrimesPerPop"].std()

maximum_violent_crime = df["ViolentCrimesPerPop"].max()

minimum_violent_crime = df["ViolentCrimesPerPop"].min()

#this will print the statements for the statistical categorys
print(f"The mean for violent crimes per population is: {average_violent_crime:.2f}")

print(f"The median for violent crimes per population is: {median_violent_crime:.2f}")

print(f"The standard deviation crimes per population is: {standard_deviation_violent_crime:.2f}")

print(f"The maximum violent crime value per population is: {maximum_violent_crime}")

print(f"The minimum violent crime value per population is: {minimum_violent_crime}")


# the mean > median so right skewed

# If there are extreme values the mean would be effected alot since its calculation is the summing of every data point
