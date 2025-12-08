
import numpy as np

A, k, c = 10, 2, 0.05


#function
def g(delta):

    return A * np.exp(-k*delta) * (1 - k*(delta - c))

#derivative of function
def g_prime(delta):
    #derivative
    return -A * k * np.exp(-k*delta) * (2 - k*(delta - c))


def newton(f, f_prime, x0, tol=1e-5, max_iterations = 10000):
    x = x0
    count = 0
    history = []

    while abs(f(x)) > tol:

        #stopping checks
        if count > max_iterations:
            raise Exception("never converged, too any iterations")
        elif abs(f_prime(x)) < tol:
            raise Exception("f' is too small, this will blow up")

        history.append(x)

        #recurrence relation
        x_new = x - f(x)/f_prime(x)
        
        x = x_new
        count += 1
        


    result = {
        'solution': x,
        'iterations': count,
        'history': history,
    }
    
    return result



if __name__ == "__main__":
    root = newton(g, g_prime, x0=0.5)
    print(f"found root: {root}")
    print(f"analytical solution: {0.05 + 1/2}")