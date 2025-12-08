

import numpy as np
import matplotlib.pyplot as plt





















def bisection(f, a, b, tol=10e-5): 

    count, history = 0, []

    while b - a > tol:
        c = (a + b) / 2
        

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

        count += 1
        history.append(c)

    result = {
        'solution': c,
        'iterations': count,
        'history': history,
    }
    
    return result



    


if __name__ == "__main__":
    x = np.linspace(-10, 10, 100) 

    y = x**3 - 10

    fx = lambda x: x**3 - 10

    print(bisection(fx, -10, 10))

    plt.plot(x, y, label='$y = x^2$') # Add a label for the legend and use TeX notation for the equation

    # 4. Add labels, a title, and a grid (optional, but recommended)
    plt.title('Graph of $y = x^2$ function')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(True)
    plt.legend() # Show the legend with the label from plt.plot()


    plt.show()