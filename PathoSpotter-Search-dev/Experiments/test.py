import numpy as np

# Save
dictionary = {'hello':'world'}
#np.save('my_file.npy', dictionary) 

import json
fold_var = 1
# Serialize data into file:
file_name = "results_k-fold"+str(fold_var)+".json"
json.dump( dictionary, open( file_name, 'w' ) )