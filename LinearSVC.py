
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style 
from sklearn import svm 
#Ulazni podaci 
x = [3,6,3,15,10,12,17,17,18,3,9,10,20,19,15,8,5,14,18,17,13,10,6,9,14,12,9,9,6]
y = [44,9,5,7,4,3,9,3,47,2,7,5,71,10,7,8,30,3,9,4,75,3,5,3,16,60,7,4,9]
#Grafički prikaz točaka 

plt.scatter(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#Priprema podataka za treniranje
X = np.array([[3,44],
              [6,9],
              [3,5],
              [15,7],
              [10,4],
              [12,3],
              [17,9],
              [17,3],
              [18,47],
              [3,2],
              [9,7],
              [10,5],
              [20,71],
              [19,10],
              [15,7],
              [8,8],
              [5,30],
              [14,3],
              [18,9],
              [17,4],
              [13,75],
              [10,3],
              [6,5],
              [9,3],
              [14,16],
              [12,60],
              [9,7],
              [9,4],
              [6,9]])

y = [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0]


#Pozivanje algoritma 
clf = svm.SVC(kernel = 'linear', C = 1.0)
clf.fit(X,y)
#Testiranje predikcije 
print ("Test_prediction 1 = " + str(clf.predict([[0.5,6.45]])))
print ("Test_prediction 2 = " + str(clf.predict([[15,19]])))

#Konstrukcija hiper ravnine za grafički prikaz 
w = clf.coef_[0]
print ("w = " + str(w))
a = -w[0]/w[1]
print ("a = " + str(a))
xx = np.linspace(0,22)
yy = a * xx  - clf.intercept_[0]/w[1]
#Grafički prikaz podataka
h0 = plt.plot(xx,yy, 'k-')
plt.scatter(X[:,0],X[:,1],c = y)
plt.grid(True)
plt.xlabel('x')
plt.xlim(0,22)
plt.ylabel('y')
plt.show()