import numpy as np 
# 9.9, -9.9ish
DS_ira = 0.0603479
SS_ira = 0.354202	
strike_ira = 203

#  124.4537 
# AFGHAN 2
DS_two = -1.38884
SS_two =  -0.952875
strike_two = 238

# AGHAN 1 77
DS_one =  -1.17731	
SS_one = 0.258781
strike_one = 278

rake_afgh_one = -np.degrees(np.arctan2(DS_one,SS_one))
rake_two = -np.degrees(np.arctan2(DS_two,SS_two))
rake_ira = -np.degrees(np.arctan2(DS_ira,SS_ira))

print(rake_ira)
print(rake_two)
print(rake_afgh_one)
