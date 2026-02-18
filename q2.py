import pandas as pd
import matplotlib.pyplot as plt

#thia loads the dataset
df = pd.read_csv("crime1.csv")

# for the histograms
plt.hist(df["ViolentCrimesPerPop"], bins=20)

plt.title("Distribution of Violent Crimes Per Population")

plt.xlabel("Violent Crimes Per Population")

plt.ylabel("Frequency")

plt.show()

#for the Box Plot
plt.figure()

plt.boxplot("ViolentCrimesPerPop")

plt.title("Box Plot of Violent Crimes Per Population")

plt.xlabel("Violent Crimes Per Population")

plt.ylabel("Values")

plt.show()



# values of ViolentCrimesPerPop are distributed and it shown by the histogram.
# it allows us to see how data is spread around different values at diff frequency ranges
# the line inside the box of the box plot is the median value and its the middle of the data set.
# The length of the box represents the interquartile range.
# the presence of  outliers in the dataset is suggested by any points outside the whiskers.

