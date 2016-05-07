'''
Applying equation y = (k1 + k2 * x) ^ k3 to solve IQ tests
Using Gradient Descent method to find optimal k1, k2 and k3

'''

# from numpy import *
import numpy as np
import gmpy2
import decimal
from decimal import Decimal as dc

# y = mx + b
# m is slope, b is y-intercept
# def compute_error_for_line_given_points(b, m, points):
def compute_error_for_line_given_points(k1, k2, k3, points):
    totalError = 0
    for i in range(0, len(points)):
        x = decimal.Decimal( points[i, 0] )
        y = decimal.Decimal( points[i, 1] )
    k1 = decimal.Decimal( k1 )
    k2 = decimal.Decimal( k2 )
    k3 = decimal.Decimal( k3 )
    totalError += ((y - (k1 + k2 * x) ** k3) ** 2)
    return totalError / decimal.Decimal( float(len(points)) )

# def step_gradient(b_current, m_current, points, learningRate):
def step_gradient(k1_current, k2_current, k3_current, points, learningRate):
    decimal.setcontext(decimal.Context(prec=10))
    k1_gradient = decimal.Decimal(0.0)
    k2_gradient = decimal.Decimal(0.0)
    k3_gradient = decimal.Decimal(0.0)
    N = decimal.Decimal( float(len(points)) )
    for i in range(0, len(points)):
        x = decimal.Decimal( points[i, 0] )
        y = decimal.Decimal( points[i, 1] )
        
        try:
            common_part_1 = -(2/N) * (y - (k1_current + k2_current * x) ** k3_current)
        except Exception as e:
            print('Exception #1')
            print('(y - (k1_current + k2_current * x)) : ', (y - (k1_current + k2_current * x)))
            print('k3_current : ', k3_current)
            
        try:
            common_part_2 = (k3_current * (k1_current + k2_current * x) ** (k3_current - 1))
        except Exception as e:
            print('Exception #2')
            print('x : ', x)
            print('k3_current - 1 : ', (k3_current - 1))
            print('k3_current * (k1_current + k2_current * x) : ', (k3_current * (k1_current + k2_current * x)))
            
        try:
            common_part_3 = decimal.Decimal.ln( k1_current + k2_current * x)
        except Exception as e:
            print('Exception #3')
            print('x : ', x)
            print('k1_current + k2_current * x : ', (k1_current + k2_current * x))
            
        try:
            common_part_4 = (k1_current + k2_current * x) ** k3_current
        except Exception as e:
            print('Exception #4')
            print('x : ', x)
            print('k3_current : ', k3_current)
            print('k1_current + k2_current * x : ', (k1_current + k2_current * x))
        
#         common_part_1 = -(2/N) * (y - (k1_current + k2_current * x) ** k3_current)
#         common_part_2 = (k3_current * (k1_current + k2_current * x) ** (k3_current - 1))
#         common_part_3 = decimal.Decimal.ln( k1_current + k2_current * x)
#         common_part_4 = (k1_current + k2_current * x) ** k3_current
        k1_gradient += common_part_1 * common_part_2
        k2_gradient += common_part_1 * common_part_2 * x
        k3_gradient += common_part_1 * common_part_3 * common_part_4
        
#         k1_gradient += -(2/N) * (y - (k1_current + k2_current * x) ** k3_current) * (k3_current * (k1_current + k2_current * x) ** (k3_current - 1))
#         k2_gradient += -(2/N) * (y - (k1_current + k2_current * x) ** k3_current) * (k3_current * (k1_current + k2_current * x) ** (k3_current - 1)) * x
#         k3_gradient += -(2/N) * (y - (k1_current + k2_current * x) ** k3_current) * np.log(k1_current + k2_current * x) * (k1_current + k2_current * x) ** k3_current
        if(dc.is_nan(k1_gradient) or dc.is_nan(k2_gradient) or dc.is_nan(k1_gradient) or
           dc.is_infinite(k1_gradient) or dc.is_infinite(k2_gradient) or dc.is_infinite(k1_gradient)): 
            print('   i=', i)
            print('   x = ', x, ',  y = ', y)
            print('   comPart_2_1 = ', (k3_current * (k1_current + k2_current * x)))
            print('   comPart_2_2 = ', (k3_current - 1))
            print('   comPart_2_3 = ', k3_current * (k1_current + k2_current * x) ** (k3_current - 1))
            print('   k1_cur=', k1_current, '  k2_cur=', k2_current, '  k3_cur=', k3_current)
            print('   common_part_1 = ', common_part_1)
            print('   common_part_2 = ', common_part_2)
            print('   common_part_3 = ', common_part_3)
            print('   k1_gradient = ', k1_gradient)
            print('   k2_gradient = ', k2_gradient) 
            print('   k3_gradient = ', k3_gradient, '\n')
            k1_gradient = 1001.0
            break
        
    new_k1 = k1_current - (learningRate * k1_gradient)
    new_k2 = k2_current - (learningRate * k2_gradient)
    new_k3 = k3_current - (learningRate * k3_gradient)
    return [new_k1, new_k2, new_k3, k1_gradient, k2_gradient, k3_gradient]


# def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
def gradient_descent_runner(points, starting_k1, starting_k2, starting_k3, learning_rate, num_iterations):
    k1 = decimal.Decimal( starting_k1 )
    k2 = decimal.Decimal( starting_k2 )
    k3 = decimal.Decimal( starting_k3 )
    learning_rate = decimal.Decimal( learning_rate )
    k1_r = 0.0
    k2_r = 0.0
    k3_r = 0.0
    c = 0
    error = 10000.0
    for i in range(num_iterations):
        k1, k2, k3, k1_g, k2_g, k3_g = step_gradient(k1, k2, k3, np.array(points), learning_rate)
#         if(error > compute_error_for_line_given_points(k1, k2, k3, array(points))):
#             error = compute_error_for_line_given_points(k1, k2, k3, array(points))
        if(k1_g == 1001.0):
            break
        k1_r = k1
        k2_r = k2
        k3_r = k3
        c = c + 1
        if(c % 10000 == 0):
            print("C: ", c , '  k1: ',k1, '  k2: ',k2, '  k3: ',k3, '  k1_g: ',k1_g, '  k2_g: ',k2_g, '  k3_g: ',k3_g)
    return [k1_r, k2_r, k3_r, c]

def run():
    points = np.genfromtxt("E:/Eq_y=-2+1x.csv", delimiter=";")
    for i in range(9):
        print(points[i][0], points[i][1])
    learning_rate = 0.01
    initial_k1 = 0.1
    initial_k2 = 0.1
    initial_k3 = 0.1
    num_iterations = 500000
    print ("Starting gradient descent at k1 = {0}, k2 = {1}, k3 = {2}, error = {3}".format(initial_k1, initial_k2, initial_k3, compute_error_for_line_given_points(initial_k1, initial_k2, initial_k3, points)))
    print ("Running...")
    [k1, k2, k3, num_iterations] = gradient_descent_runner(points, initial_k1, initial_k2, initial_k3, learning_rate, num_iterations)
    print ("After {0} iterations k1 = {1}, k2 = {2}, k3 = {3}, error = {4}".format(num_iterations, k1, k2, k3, compute_error_for_line_given_points(k1, k2, k3, points)))

if __name__ == '__main__':
    run()