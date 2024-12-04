import pygmt 
import numpy as np 
np.random.seed(42)
region=[150,240,-10,60]
x=np.random.uniform(region[0],region[1],100)
y=np.random.uniform(region[2],region[3],100)
fig = pygmt.Figure()
fig.basemap(region=region,projection='X6i',frame=True)
fig.plot(x=x,y=y,style='i0.5c',fill='black')
fig.show()

