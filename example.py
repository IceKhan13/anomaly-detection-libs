import numpy as np
from lib.ldcof import LDCOF


if __name__ == '__main__':
    x = np.random.random((20, 3))
    x_train, x_test = x[0:13], x[13:]
    ldcof = LDCOF()
    ldcof.fit(x_train)

    ldcof.transform(x_test)
