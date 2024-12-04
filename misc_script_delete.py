import numpy as np 

files = ['20230926_20231101_1s.txt',
         '20230926_20231020_1s.txt',
         '20230902_20231101_1s.txt',
         '20230902_20231020_1s.txt'
]

ones = np.zeros(1835) + 1
for file in files:
    f = open(file, "w")
    for ii in range(len(ones)):
        string = (str(ones[ii]) + '\n')      
        f.write(string)
    f.close()

