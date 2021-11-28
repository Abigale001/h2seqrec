import pickle
import numpy 

# to avoid the pickle version problem
file_name = './output_embed/pretrain-AMT.2021.01.25.14.04.20.pkl'
out_file_name = './output_embed/pretrain-AMT.2021.01.25.dim.100.pkl'

embeds = pickle.load(open(file_name, "rb"))
pickle.dump(embeds, open(out_file_name, 'wb'), protocol=4)