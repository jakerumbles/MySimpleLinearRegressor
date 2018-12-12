"""
Author: Jake Edwards
Date: 6/20/2018
Filename: MySimpleLinearRegressor.py
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MySimpleLinearRegressor():
    """Simple Linear Regressor:
        1) First create the regressor object passing in a filepath for your .csv file
        2) Call the calculate() method on the regressor object to compute the regression line
        3) Call the plot() method on the regressor object to print the graph of observations
           and the regression line.
           
        Variables:
            observations - This is the csv file passed to the constructer.  It should contain
            only two rows, x and y.
            
            df - This is a pandas DataFrame for holding other information needed in the calculations.
            It will be concatenated with the table DataFrame.
            
            table - The is the concatenation of the observations table and the df table.  It holds
            the imported observations from the .csv file and all other relevant pieces of information
            to complete the calculations.
            
            a - The y-intercept of the regression line.
            
            b - The slope of the regression line.
           """
         
    
    def __init__(self, filepath):
        """Reads in the .csv file and stores it in a DataFrame called observations. It is then
        concatenated with the df DataFrame to create the table DataFrame"""
        # Read in csv file
        self.observations = pd.read_csv(filepath)
        # Create other DataFrame needed
        self.df = pd.DataFrame(
        {"(x-xbar)" : ['nan'],
         "(y-ybar)" : ['nan'],
         "(x-xbar)*(y-ybar)" : ['nan'],
         "(x-xbar)**2" : ['nan'],
         "(y-ybar)**2" : ['nan']},
         index = [i for i in range(len(self.observations))])

        # Concatenate observations with df self.table
        self.table = pd.concat([self.observations, self.df], axis=1)
        self.a = 0
        self.b = 0
        
        # Disable pandas warning
        pd.options.mode.chained_assignment = None  # default='warn'
        
    def calculate(self):
        """Performs all of the calculations to find the regression line"""
        # Compute (x-xbar) and fill in the respective column of self.table DataFrame
        # First compute xbar from x column in self.table
        xbar = self.table['x'].mean()
        
        counter = 0
        for x in self.table['x']:
            self.table['(x-xbar)'][counter] = x - xbar
            counter += 1
        # (x-xbar) now calculated
        
        # Compute ybar, then compute (y-ybar) and fill in respective column of self.table DataFrame
        ybar = self.table['y'].mean()
        
        counter = 0
        for y in self.table['y']:
            self.table['(y-ybar)'][counter] = y - ybar
            counter += 1
        # (y-ybar now calculated)
        
        # Calculate (x-xbar)*(y-ybar) and fill in respective column of self.table DataFrame
        for i in range(len(self.table)):
            self.table['(x-xbar)*(y-ybar)'][i] = self.table['(x-xbar)'][i] * self.table['(y-ybar)'][i]
        # (x-xbar)*(y-ybar) now calculated
        
        # Calculate (x-xbar)**2 and fill in respective column of self.table Dataframe
        counter = 0
        for num in self.table['(x-xbar)']:
            self.table['(x-xbar)**2'][counter] = num**2
            counter += 1
        # (x-xbar)**2 now calculated
        
        # Calculate (y-ybar)**2 and fill in respective column of self.table DataFrame
        counter = 0
        for num in self.table['(y-ybar)']:
            self.table['(y-ybar)**2'][counter] = num**2
            counter += 1
        # (y-ybar)**2 now calculated
        
        
        """
        The self.table is fully populated and now we can calculate the Linear Regression
        Function that fits the provided oberservations in the self.table DataFrame.
        
        Linear Regression Formula: y = a + bx
        
        Slope(b) of Regression Line: b = r(std. deviation(y)/std. deviation(x))
            where r is Pearson's Correllation Coefficient
            r = SUM((x-xbar)*(y-ybar)) / SQRT(SUM(x-xbar)**2 * SUM(y-ybar)**2)
        
        y-intercept of Regression Line: a = ybar - b(xbar)
        """
        
        # First calculate Pearson's Correllation Coefficient(r)
        # Calculate numerator
        numerator = sum(self.table['(x-xbar)*(y-ybar)'])
        
        # Calculate denominator
        sum1 = sum(self.table['(x-xbar)**2'])
        sum2 = sum(self.table['(y-ybar)**2'])
        
        r = numerator / (sum1 * sum2)**.5 # Pearson's Correllation Coefficient
        
        # Calculate the slope of the Regression Line(b)
        y_std = self.table['y'].std()
        x_std = self.table['x'].std()
        self.b = r * (y_std/x_std) # Slope of the Regression Line
        
        # Finally calculate the y-intercept(a) of the Regression Line
        self.a = ybar - (self.b * xbar)

    def plot(self):
        """Plots the graph with the real observations and the regression line"""
        # Now let's plot the observations and the the Regression Line using the above calculated formula
        plt.scatter(self.table.x, self.table.y) # Scatter plot of the observations
        #x_grid = np.arange()
        plt.plot(self.table.x, self.a + (self.b * self.table.x))
        plt.title('Simple Linear Regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

