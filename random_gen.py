import numpy as np

with open('output/random_out.csv', 'w') as f:
    for i in range(1, 2382):
        f.write(str(i))
        f.write(',')
        aa = np.random.random()
        b = 0
        if aa <= 0.25:
            b = 3
        elif aa <= 0.5:
            b = 4
        elif aa <= 0.7:
            b =6
        elif aa <= 0.775:
            b=7
        elif aa <= 0.825:
            b = 5
        elif aa <= 0.875:
            b = 8
        elif aa <= 0.925:
            b = 10
        elif aa <= 0.95:
            b = 11
        elif aa <= 0.975:
            b = 2
        elif aa <= 1:
            b = 9
        f.write(str(b))
        f.write('\n')

