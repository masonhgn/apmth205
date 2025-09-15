
import numpy as np


def q2_1() -> None:
    table = {}
    for exp in range(2,8):
        pn = np.prod([1 + 1/j for j in range(1,10**exp+1)])

        table[exp] = pn

    for key, val in table.items():
        print(f'10^{key}: {val}')



if __name__ == "__main__":
	q2_1()
