import wisardpkg as wp
import numpy as np
# load input data, just zeros and ones
X = np.array([
      [1,1,1,0,0,0,0,0],
      [1,1,1,1,0,0,0,0],
      [0,0,0,0,1,1,1,1],
      [0,0,0,0,0,1,1,1]
    ])

# load label data, which must be a string array
y = [
      "cold",
      "cold",
      "hot",
      "hot"
    ]


addressSize = 3     # number of addressing bits in the ram
ignoreZero = False # optional; causes the rams to ignore the address 0

# False by default for performance reasons,
# when True, WiSARD prints the progress of train() and classify()
verbose = False

wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)



# train using the input data
wsd.train(X,y)

# classify some data
out = wsd.classify(X)

# the output of classify is a string list in the same sequence as the input
for i,d in enumerate(X):
    print(out[i],d)