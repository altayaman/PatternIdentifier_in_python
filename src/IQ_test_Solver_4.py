'''
Applying equation y = k1(x) + k2(x^2) + k2(x^3) to solve IQ tests
Using Gradient Descent method to find optimal k1, k2 and k3

'''

from numpy import *
import gmpy2
import decimal

def compute_error_for_line_given_points(k1, k2, k3, points):
    totalError = 0
    for i in range(0, len(points)):
        x = decimal.Decimal( points[i, 0] )
        y = decimal.Decimal( points[i, 1] )
    k1 = decimal.Decimal( k1 )
    k2 = decimal.Decimal( k2 )
    k3 = decimal.Decimal( k3 )
    y_hat = k1 * x + k2 * (x**2) + k3 * (x**3)
    totalError += (y - y_hat) ** 2
    return totalError / decimal.Decimal( float(len(points)) )

def step_gradient(k1_current, k2_current, k3_current, points, learningRate):
#     gmpy2.set_context(gmpy2.context(precision=10))
    decimal.setcontext(decimal.Context(prec=10))
    k1_gradient = decimal.Decimal(0.0)
    k2_gradient = decimal.Decimal(0.0)
    k3_gradient = decimal.Decimal(0.0)
    N = decimal.Decimal( float(len(points)) )
    for i in range(0, len(points)):
        x = decimal.Decimal( points[i, 0] )
        y = decimal.Decimal( points[i, 1] )
        y_hat = decimal.Decimal( k1_current * x + k2_current * (x**2) + k3_current * (x**3) )
        k1_gradient += decimal.Decimal( -(2/N) * (y - y_hat) * (x) )
        k2_gradient += decimal.Decimal( -(2/N) * (y - y_hat) * (x ** 2) )
        k3_gradient += decimal.Decimal( -(2/N) * (y - y_hat) * (x ** 3) )
    new_k1 = k1_current - (learningRate * k1_gradient)
    new_k2 = k2_current - (learningRate * k2_gradient)
    new_k3 = k3_current - (learningRate * k3_gradient)
    return [new_k1, new_k2, new_k3, k1_gradient, k2_gradient, k3_gradient]

def gradient_descent_runner(points, starting_k1, starting_k2, starting_k3, learning_rate, num_iterations):
#     k1 = gmpy2.mpz(starting_k1)
#     k2 = gmpy2.mpz(starting_k2)
#     k3 = gmpy2.mpz(starting_k3)
    k1 = decimal.Decimal( starting_k1 )
    k2 = decimal.Decimal( starting_k2 )
    k3 = decimal.Decimal( starting_k3 )
    learning_rate = decimal.Decimal( learning_rate )
    k1_r = 0.0
    k2_r = 0.0
    k3_r = 0.0
    c = 0
    error = 10000.0
    for i in range(len(points[:, 0])):
        points[i][1] = points[i][1] / max(points[:,1])
    for i in range(num_iterations):
        k1, k2, k3, k1_g, k2_g, k3_g = step_gradient(k1, k2, k3, array(points), learning_rate)
#         if(error > compute_error_for_line_given_points(k1, k2, k3, array(points))):
#             error = compute_error_for_line_given_points(k1, k2, k3, array(points))
        k1_r = k1
        k2_r = k2
        k3_r = k3
        c = c + 1
        if(c % 1000 == 0):
            print("C: ", c , '  k1: ',k1, '  k2: ',k2, '  k3: ',k3, '  k1_g: ',k1_g, '  k2_g: ',k2_g, '  k3_g: ',k3_g)
    return [k1_r, k2_r, k3_r, c]

def run():
    points = genfromtxt("E:/Eq_y=x.csv", delimiter=";")
    for i in range(9):
        print(points[i][0], points[i][1])
    learning_rate = 0.00001
    initial_k1 = 1
    initial_k2 = 0
    initial_k3 = 0
    num_iterations = 10000
    print ("Starting gradient descent at k1 = {0}, k2 = {1}, k3 = {2}, error = {3}".format(initial_k1, initial_k2, initial_k3, compute_error_for_line_given_points(initial_k1, initial_k2, initial_k3, points)))
    print ("Running...")
    [k1, k2, k3, num_iterations] = gradient_descent_runner(points, initial_k1, initial_k2, initial_k3, learning_rate, num_iterations)
#     print ("After {0} iterations k1 = {1}, k2 = {2}, k3 = {3}, error = {4}".format(num_iterations, k1, k2, k3, compute_error_for_line_given_points(k1, k2, k3, points)))
    print ("After {0} iterations".format(num_iterations))
    print('k1 = ', k1)
    print('k2 = ', k2)
    print('k3 = ', k3)
    print('error = ', compute_error_for_line_given_points(k1, k2, k3, points))

if __name__ == '__main__':
    run()