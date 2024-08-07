import numpy as np
from scipy.optimize import curve_fit

def FitChi2Pol2(x, y, chi2_level=1., return_param = False):
  a,b,c = np.polyfit(x,y,2)
  xmin = -b/2./a
  fmin = a*xmin*xmin + b*xmin + c
  c1 = c - chi2_level -fmin
  x1 = (-b + np.sqrt(b*b-4*a*c1))/(2*a)
  x2 = (-b - np.sqrt(b*b-4*a*c1))/(2*a)
  dx12 = abs(x1-x2)
  x_min = 0.5*(x1+x2)
  if return_param == False:
    return dx12/x_min*100./2.
  else:
    return dx12/x_min*100./2., a, b, c



def FitChi2Pol2_through_zero(x, y, p0, chi2_level=1., return_param = False):
  # c = b^2/4a in this case
  f = lambda x, a, b: a*x**2 + b*x + b**2/4./a
  (a,b),_ = curve_fit(f,x,y,p0=p0)
  c = b**2/(4*a)
  c1 = c - chi2_level
  x1 = (-b + np.sqrt(b*b-4*a*c1))/(2*a)
  x2 = (-b - np.sqrt(b*b-4*a*c1))/(2*a)
  dx12 = abs(x1-x2)
  x_min = 0.5*(x1+x2)
  if return_param == False:
    return dx12/x_min*100./2.
  else:
    #return x_min, a, b, c
    return dx12/x_min*100./2., a, b, c
 


def Scan():
  # example
  arr = np.random.random([4,5,2,6])
  for idx, value in np.ndenumerate(arr):
    print(idx, value)
