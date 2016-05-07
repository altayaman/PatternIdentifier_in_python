import numpy as np
import gmpy2
import decimal


x = np.array([1,2,3,4])
print(type(x))

print('Ln(3) = ',np.log(3))

gmpy2.set_context(gmpy2.context(precision=200))
# a = gmpy2.sqrt(5)
# print('a = ', a)
# print('a^2 = ', a ** 2)
# print('a^-2.33333333333 = ', a ** (-2.33333333333))
b = decimal.Decimal(-5.6212117463847590438596596507406976874)
# b = gmpy2.mpz(5.0000000000000000000000000004)
print('b = ', b ** 2, ', ', round(b, 0))

decimal.setcontext(decimal.Context(prec=40))
d=decimal.Decimal(5)#.sqrt()
print('d = ', d)
dd = 2 * d
print('dd type : ', type(dd))
print(2 ** -3.837456387643573485743897589437589437)