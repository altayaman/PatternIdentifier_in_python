'''
Applying the following algorithms to solve IQ tests:
 - Random Forest Classifier
 - Random Forest Regressor
 - SVM
 - Linear Model

'''


import numpy as np
import pandas as pn
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import xlrd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from scipy.interpolate import *
import csv as csv
import xlsxwriter
from numpy import polyfit, poly1d
from scipy import stats


# ------  Read 1st sheet data from excel file  ------------------------
xls_file = 'E:/IQtest_2 - y=x^2.xlsx';
xls = pn.ExcelFile(xls_file)
# print('Excel file sheets: ', xls.sheet_names)

df = xls.parse('Sheet1')
# print(df['Order'][:5])

# ------  Convert data into array  -------------------------------------
# ----------------------------------------------------------------------
dataSetArray =np.array(df) 

# -------  Create X and Y for training  --------------------------------
# ----------------------------------------------------------------------
X_train = dataSetArray[[0,1,2,3,4], 0]
Y_train = dataSetArray[[0,1,2,3,4], 1]
X_train = X_train.reshape(5,1)        # Reshape array into [5,1] because 1d array will be interpreted as single sample
# print(type(X_train))  # check the datatype
# print(Y_train.shape)  # check the shape of numpy array

# -------  Create X and Y for testing  ---------------------------------
# ----------------------------------------------------------------------
X_test = dataSetArray[[5,6,7,8], 0]
Y_test = dataSetArray[[5,6,7,8], 1]
X_test = X_test.reshape(4,1) 

#-------------  Linear model ------------------------
# ----------------------------------------------------------
print('\nLinear Regression Training...')
Lin_Reg = linear_model.LinearRegression()
Lin_Reg.fit(X_train, Y_train)

slope, intercept, r_value, p_value, std_err = stats.linregress(dataSetArray[[0,1,2,3,4], 0], dataSetArray[[0,1,2,3,4], 1])
print('intercept = ',intercept)
print('slope = ', slope)


print('Linear Regression Predicting...')
Lin_Reg_prediction = Lin_Reg.predict(X_test)#.astype(np.int32)

print('Quality of Linear Regression prediction ', np.sum(Y_test==Lin_Reg.predict(X_test)) / float(len(X_test)), '%')

# plt.scatter(X_test, Y_test,  color='black')
# plt.plot(X_test, Lin_Reg.predict(X_test), color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
#  
# plt.show()

#-------------  Polynomial model ------------------------
# ----------------------------------------------------------
print('\nPolynomial Regression Training...')
degree = 2
poly_1_coeffs = polyfit(df['Order'][:5], df['Result'][:5], degree)
poly_prediction = poly1d(poly_1_coeffs)


print("Polynomail coeffs: ", poly_1_coeffs)

print('Linear Regression Predicting...')
Poly_prediction = poly_prediction(X_test)#.astype(np.int32)

print('Quality of Linear Regression prediction ', np.sum(Y_test==poly_prediction(X_test)) / float(len(X_test)), '%')


# plt.scatter(X_test, Y_test,  color='black')
# plt.plot(X_test, Lin_Reg.predict(X_test), color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
#  
# plt.show()

#-------------  Random Forest Classifier model ------------------------
# ----------------------------------------------------------
print('\nRF Classifier Training...')
RF_classifier = RandomForestClassifier(n_estimators=100, n_jobs=2)
RF_classifier.fit( X_train, Y_train)
print('RF Classifier Predicting...')
RF_clf_prediction = RF_classifier.predict(X_test)#.astype(int)
  
print('Quality of Random Forest prediction ', np.sum(Y_test==RF_classifier.predict(X_test)) / float(len(X_test)), '%')

# plt.scatter(X_test, Y_test,  color='black')
# plt.plot(X_test, RF.predict(X_test), color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# 
# plt.show()

#-------------  Random Forest Regressor model ------------------------
# ----------------------------------------------------------

print('\nRF Classifier Training...')
RF_regressor = RandomForestRegressor(n_estimators=100, n_jobs=2, min_samples_split=1)
RF_regressor.fit( X_train, Y_train)
print('RF Classifier Predicting...')
RF_regr_prediction = RF_regressor.predict(X_test)#.astype(int)
  
print('Quality of Random Forest prediction ', np.sum(Y_test==RF_regressor.predict(X_test)) / float(len(X_test)), '%')

#-------------  SVM model  --------------------------------
# ---------------------------------------------------------
print('\nSVM Training...')
clf = svm.SVC(gamma = 0.001, C = 100, kernel='rbf')
clf.fit(X_train, Y_train)

print('SVM Predicting...')
SVM_prediction = clf.predict(X_test) #.astype(int)
  
print('Quality of SVM prediction ', np.sum(Y_test==clf.predict(X_test)) / float(len(X_test)), '%')

# print(pn.crosstab(Y_test, SVM_prediction))

# -------  Write Linear prediction into XLS  -----------------------------------
# **************************************************************************
df = pn.DataFrame(Lin_Reg_prediction, Y_test)
df.index.name = 'Expected'
df.columns = ['Predicted']

writer = pn.ExcelWriter('E:/IQtest_prediction_Lin_Reg_XLS1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

# ******  END  ************************************************************

# -------  Write Linear prediction into XLS  -----------------------------------
# **************************************************************************
df = pn.DataFrame(Poly_prediction, Y_test)
df.index.name = 'Expected'
df.columns = ['Predicted']

writer = pn.ExcelWriter('E:/IQtest_prediction_Poly_Reg_XLS1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

# ******  END  ************************************************************

# -------  Write RF classifier prediction into XLS  -----------------------------------
# **************************************************************************
df = pn.DataFrame(RF_clf_prediction, Y_test)
df.index.name = 'Expected'
df.columns = ['Predicted']

writer = pn.ExcelWriter('E:/IQtest_prediction_RF_clf_XLS1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

# ******  END  ************************************************************

# -------  Write RF regressor prediction into XLS  -----------------------------------
# **************************************************************************
df = pn.DataFrame(RF_regr_prediction, Y_test)
df.index.name = 'Expected'
df.columns = ['Predicted']

writer = pn.ExcelWriter('E:/IQtest_prediction_RF_regr_XLS1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

# ******  END  ************************************************************

# -------  Write SVM prediction into XLS - Option 1  -----------------------
# **************************************************************************
df = pn.DataFrame(SVM_prediction, Y_test)
df.index.name = 'Expected'
df.columns = ['Predicted']

writer = pn.ExcelWriter('E:/IQtest_prediction_SVM_XLS1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

# ******  END  ************************************************************

plt.plot(X_test, Y_test, label="ground truth")
plt.scatter(X_test, Y_test, label="training points")
plt.plot(X_test, Poly_prediction, label="degree %d" % degree)
plt.legend(loc='upper left')
plt.show()

# -------  Concatenate Y_test and predictions into one array    ------------
# Y_test = Y_test.reshape(4,1)
# SVM_prediction = SVM_prediction.reshape(4,1)
# dd = np.concatenate((Y_test, SVM_prediction), axis = 1)


# -------  Write SVM prediction into XLS - Option 2  -----------------------
# **************************************************************************

# -------  Step - 1: Create XLS writer    -----------------------
# workbook = xlsxwriter.Workbook('E:/IQtest_SVM_prediction_XLS2.xlsx')
# worksheet = workbook.add_worksheet()

# -------  Step - 2: Write data  --------------------------------
# row = 0
# 
# for col, data in enumerate(dd):
#     worksheet.write_column(row, col, data)

# -------  Step - 3: Close XLS writer  ---------------------------
# workbook.close()

# ******  END  ************************************************************



# -------  Write SVM prediction into CSV - Option 1  -----------------------
# **************************************************************************

# -------  Step - 1: Create CSV writer    -----------------------
# predictions_file = open("E:/IQtest_SVM_prediction_CSV1.csv", "w")
# open_file_object = csv.writer(predictions_file)


# -------  Step - 2: Write data  --------------------------------
# for values in dd:
#     open_file_object.writerow(values)
    
# -------  Step - 3: Close CSV writer  ---------------------------
# predictions_file.close()
# print("Results File created ... ")

# ******  END  ************************************************************



# -------  Write SVM prediction into CSV - Option 2  -----------------------
# **************************************************************************

# -------  Step - 1: Create CSV writer    -----------------------
# predictions_file = open("E:/IQtest_SVM_prediction_CSV2.csv", "w")
# open_file_object = csv.writer(predictions_file)


# -------  Step - 2: Write data  --------------------------------
# # open_file_object.writerow(["Order","Result"])
# # open_file_object.writerows(zip(Y_sample_30, SVM_prediction))
# for i in range(0,len(SVM_prediction)):
#     open_file_object.writerow([Y_test[i], SVM_prediction[i]])

# -------  Step - 3: Close CSV writer  --------------------------
# predictions_file.close()
# print("Results File created ... ")

# ******  END  ************************************************************

