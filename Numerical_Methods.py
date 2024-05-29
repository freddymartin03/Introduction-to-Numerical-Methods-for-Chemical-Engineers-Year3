import numpy as np
import matplotlib.pyplot as plt


class LinearEquations:
    def __init__(self, A, b, x0, tol, max_iter, w):
        '''
        A: coefficient matrix
        b: right-hand side vector
        x0: initial guess vector
        tol: tolerance for convergence
        maxIter: maximum number of iterations
        '''
        self.A = A
        self.b = b
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.w = w  
    
    def gauss_seidel(self, use_sor = False):
        '''
        Gauss Seidel method 
        '''
        A = self.A
        b = self.b 
        tol = self.tol
        max_iter = self.max_iter
        w = self.w
        x = np.copy(self.x0)
        n = len(b)
        error = tol + 1
        H = np.triu(A) - np.diag(np.diag(A))
        Linv = np.linalg.inv(A - H)
        AM = -Linv @ H
        B = Linv @ b
        k = 0
        iter = 0
        mtag = np.identity(n) * (1 - w) + w * AM

        x_matrix = np.zeros((n, max_iter + 1))
        x_matrix[:, 0] = x

        while np.sum(np.abs(error) >= tol) != 0 and k < max_iter:
            if use_sor:
                x_matrix[:, k + 1] = mtag @ x_matrix[:, k] + mtag @ B # SOR formula
            else:
                x_matrix[:, k + 1] = AM @ x_matrix[:, k] + B  # Gauss-Seidel formula
            error = x_matrix[:, k + 1] - x_matrix[:, k]  # finding error
            k += 1
            iter += 1
            
        if iter == max_iter:
            print('Warning: Maximum iterations reached without convergence.')
        else:
            print(f'Converged in {iter} iterations.')
        
        return x_matrix[:, k], iter

class ParabolicInterpolation:
    def __init__(self, xdata, ydata, x):
        '''
        xdata: Given x data 
        ydata: Given y data 
        x: Desired x point 
        '''
        self.xdata = xdata
        self.ydata = ydata 
        self.x = x

    def lagrangian_interpolation(self):
        xdata = self.xdata
        ydata = self.ydata
        n = len(xdata)
        y = 0
        for i in range(n):
            L = 1
            for j in range(n):
                if j != i:
                    L = L * (x - xdata[j]) / (xdata[i] - xdata[j])
            y = y + ydata[i] * L

        return y
    

class NonLinearEquations:
    def __init__(self, fname, xmin, xmax, max_f, dx, max_dx):
        '''
        fname: The name of the function to evaluate as a string (e.g., 'my_fun').
        xmin: The minimum x-value of the interval to search for roots.
        xmax: The maximum x-value of the interval to search for roots.
        max_f: The maximum allowed difference in function values to determine if a smaller step size is needed.
        max_dx: The maximum allowed step size in x-values.
        '''
        self.xmin = xmin
        self.fname = fname
        self.xmax = xmax 
        self.max_f = max_f
        self.dx = dx
        self.max_dx = max_dx

    def bracket_basic(self):
        x = self.xmin
        roots = []

        while x < self.xmax:
            f1 = self.fname(x)
            f2 = self.fname(x + self.dx)
            if f1 * f2 < 0:
                roots.append((x, f1))
                roots.append((x + self.dx, f2))
            x = x + self.dx
        
        for i in range(0, len(roots), 2):
            x1, f1 = roots[i]
            x2, f2 = roots[i + 1]
            print(f'Root interval: [{x1}, {x2}] with function values: f({x1})={f1}, f({x2})={f2}')

        xvec = np.linspace(self.xmin, self.xmax)
        x_vals = [r[0] for r in roots]
        y_vals = [r[1] for r in roots]
        plt.figure(1)
        plt.plot(xvec, self.fname(xvec), '-b') 
        plt.scatter(x_vals, y_vals, facecolors='none', edgecolors='r')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Function Plot with Root Intervals')
        plt.show()

    def adaptive_bracket(self):
        x = self.xmin
        step = self.max_dx
        x_limits = []
        while x < self.xmax:
            step_ok = True
            f1 = self.fname(x)
            f2 = self.fname(x + step)
            df = abs(f2 - f1)
            if df > self.max_f:
                step = step / 2
                step_ok = False
            
            if df < 0.1 * self.max_f:
                step = step * 1.5;
                step_ok = True

            if step > self.max_dx:
                step = self.max_dx
                step_ok = True
                
            if step_ok:
                if f1 * f2 < 0:
                    x_limits.append((x, f1))
                    x_limits.append((x + step, f2))
                x = x + step 

        for i in range(0, len(x_limits), 2):
            x1, f1 = x_limits[i]
            x2, f2 = x_limits[i + 1]
            print(f'Root interval: [{x1}, {x2}] with function values: f({x1})={f1}, f({x2})={f2}')

        xvec2 = np.linspace(self.xmin, self.xmax)
        x_vals2 = [kk[0] for kk in x_limits]
        y_vals2 = [kk[1] for kk in x_limits]
        plt.figure(2)
        plt.plot(xvec2, self.fname(xvec2), '-b') 
        plt.scatter(x_vals2, y_vals2, facecolors='none', edgecolors='r')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Function Plot with Root Intervals ADAPTIVE')
        plt.show()



'''
########################
# Gauss Seidel Example #
########################

A = np.array([[6, 3, 0], [1, 4, -1.5], [0, 1, 3]])
b = np.array([1, 1, 1])
x0 = np.zeros_like(b)
tol = 1e-6
max_iter = 1000
w = 1

solver = LinearEquations(A, b, x0, tol, max_iter, w)
solution, iterations = solver.gauss_seidel()
print('Gauss-Seidel Solution:', solution)

solution_sor, iterations_sor = solver.gauss_seidel(use_sor=True)
print('Gauss-Seidel with SOR Solution:', solution_sor)


#############
# Parabolic #
#############

xdata = [6, 3, 0]
ydata = [4, 7, 0]
x = 5

solver2 = ParabolicInterpolation(xdata, ydata, x)
y = solver2.lagrangian_interpolation()
print(y)
'''

###############################
# Bracketing Basic & Adaptive #
###############################


def my_fun(x):
    return 10 * x**3 - 7 * x**2 - 2 * x + np.exp(x / 2)

xmin = -1
xmax = 1 
max_f = 0.5
dx = 0.1 
max_dx = 0.5

solver3 = NonLinearEquations(my_fun, xmin, xmax, max_f, dx, max_dx)
roots = solver3.bracket_basic()
roots2 = solver3.adaptive_bracket()

