import os
from minisom import MiniSom
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from scipy import stats
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


data_path = '/net/fs04/d2/xgao/SOM'


som_shape = (3, 4)
x = som_shape[0]
y = som_shape[1]


# Change the variables here
var_list = ['slp']
var_combo = 'slp'


output_path = '../results/'+var_combo+'/'+str(x)+'x'+str(y)+'/'



def get_var(prod):
    dfs = []
    for var in var_list:
        df = pd.read_csv(data_path+'/'+prod+'/'+prod+'_daily_8019_SCUS_JFMAMOND_'+var+'_anom.txt', header = None, delim_whitespace=True)
        dfs.append(df)
    return dfs



def concat_df(df_list):
    comb_df = pd.concat(df_list, axis = 1)
    comb = comb_df.values
    return comb



def num_days_node(winmap):
    for cell in sorted(winmap.keys()):
        print(cell, len(winmap[cell]))




def get_winmap(data, som, return_indices = True):
    winmap = defaultdict(list)
    for i, j in enumerate(data):
        winmap[som.winner(j)].append(i+1 if return_indices else j)
    return winmap




def get_node_composites(som, data):
    winmap_patt = som.win_map(data)
    composites = {}
    for node in sorted(winmap_patt.keys()):
        composites[node] = np.mean(np.array(winmap_patt[node]), axis = 0)
    return composites




def save_day_list(winmap, prod, save = True):
    if save:        
        directory = output_path + prod
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+'/'+prod+'_node_assignment.txt', 'w') as f:
            for item in sorted(winmap.items()):
                item = str(item)[1:-1]
                f.write("%s\n\n" % item)
        f.close()
    
    else:
        print("Save is False")



def save_composites(prod, composites, save = True):
    if save:
        directory = output_path + prod
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory+'/'+prod+'_composites.json', 'wb') as f:
            pickle.dump(composites, f)
            f.close()
    else:
        print("Save is Fasle")

def perc_common(winmap, comp):
    common_list = prod_common[comp]
    print(comp)
    for i, cell in enumerate(sorted(winmap.keys())):
        num_days = len(winmap[cell])
        common = common_list[i]
        perc = (common/num_days)*100
        print(perc)


# for 3x4 or 4x3 figsize = ((15,12))
def plot_som(var, composites, prod):
    plt.tight_layout()
    fig, axs = plt.subplots(nrows = som_shape[0], ncols = som_shape[1], figsize = (15,8))
    fig.suptitle(prod+' '+var, fontsize=25, x=0.4, y=.92, horizontalalignment='left', verticalalignment='top',)

    index = var_list.index(var)
    for ax, node in zip(axs.flat, composites.keys()):
        im = ax.imshow(composites[node].reshape((len(var_list),41,49))[index], interpolation='nearest', cmap='hot')
    
    
    
    fig.subplots_adjust(top=0.88)
    
    plt.colorbar(im, ax=axs.ravel().tolist())

    directory = output_path + prod
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(directory+'/'+prod+'_heatmap.png')
    # plt.show()


#changed sigma from 0.5 and learning_rate from 0.5

#1, 0.01

sigma = 1
learning_rate = 0.01
neighbourhood_function = 'gaussian'
random_seed = 10
# iterations = 5000
# verbose = True


# In[16]:


dfs_ERA5 = get_var('ERA5')

ERA5 = concat_df(dfs_ERA5)
data_ERA5 = np.copy(ERA5)
np.random.seed(10)
# np.random.shuffle(data_ERA5)

input_len = data_ERA5.shape[1]
iterations = len(data_ERA5)*10

som_ERA5 = MiniSom(x , y, input_len, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function=neighbourhood_function, random_seed=random_seed)
# som_ERA5.train_random(data_ERA5, iterations, verbose=True)
som_ERA5.train(data_ERA5, iterations, verbose=False)



winmap_ERA5 = get_winmap(ERA5, som_ERA5)



composites_ERA5 = get_node_composites(som_ERA5, data_ERA5)



save_composites('ERA5', composites_ERA5)


save_day_list(winmap_ERA5, 'ERA5')


# MERRA2
dfs_MERRA2 = get_var('MERRA2')

MERRA2 = concat_df(dfs_MERRA2)
data_MERRA2 = np.copy(MERRA2)
np.random.seed(10)
# np.random.shuffle(data_MERRA2)

input_len = data_MERRA2.shape[1]
iterations = len(data_MERRA2)*10

som_MERRA2 = MiniSom(x , y, input_len, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function=neighbourhood_function, random_seed=random_seed)
# som_MERRA2.train_random(data_MERRA2, iterations, verbose=True)
som_MERRA2.train(data_MERRA2, iterations, verbose=False)


winmap_MERRA2 = get_winmap(MERRA2, som_MERRA2)


composites_MERRA2 = get_node_composites(som_MERRA2, data_MERRA2)


save_composites('MERRA2', composites_MERRA2)


save_day_list(winmap_MERRA2, 'MERRA2')



# NARR
dfs_NARR = get_var('NARR')

NARR = concat_df(dfs_NARR)
data_NARR = np.copy(NARR)
np.random.seed(10)
# np.random.shuffle(data_NARR)

input_len = data_NARR.shape[1]
iterations = len(data_NARR)*10

som_NARR = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function=neighbourhood_function, random_seed=random_seed)
# som_NARR.train_random(data_NARR, 5000, verbose=True)
som_NARR.train(data_NARR, iterations, verbose=False)


# In[28]:


winmap_NARR = get_winmap(NARR, som_NARR)


# In[29]:


composites_NARR = get_node_composites(som_NARR, data_NARR)


# In[30]:


save_composites('NARR', composites_NARR)


# In[31]:


save_day_list(winmap_NARR, 'NARR')


# # Number of Days Per Node

# In[32]:


num_days_node(winmap_ERA5)


# In[33]:


num_days_node(winmap_MERRA2)


# In[34]:


num_days_node(winmap_NARR)


# # Common Number of Days

# In[35]:


common_list_EM = []

for cell in sorted(winmap_ERA5.keys()):
    common = 0
    for i in winmap_ERA5[cell]:
        day = i
        if i in winmap_MERRA2[cell]:
            common += 1
    common_list_EM.append(common)


# In[36]:


common_list_EN = []

for cell in sorted(winmap_ERA5.keys()):
    common = 0
    for i in winmap_ERA5[cell]:
        day = i
        if i in winmap_NARR[cell]:
            common += 1
    common_list_EN.append(common)


# In[37]:


common_list_MN = []

for cell in sorted(winmap_MERRA2.keys()):
    common = 0
    for i in winmap_MERRA2[cell]:
        day = i
        if i in winmap_NARR[cell]:
            common += 1
    common_list_MN.append(common)


# In[38]:


common_list_EM


# In[39]:


common_list_EN


# In[40]:


common_list_MN


# In[41]:


prod_common = {
    'EM': common_list_EM,
    'EN': common_list_EN,
    'MN':common_list_MN
}


# # Percentage Common Days
perc_common(winmap_ERA5, 'EM')


# In[44]:


perc_common(winmap_ERA5, 'EN')


# In[45]:


perc_common(winmap_MERRA2, 'EM')


# In[46]:


perc_common(winmap_MERRA2, 'MN')


# In[47]:


perc_common(winmap_NARR, 'EN')


# In[48]:


perc_common(winmap_NARR, 'MN')


# # SOM Plots

# In[49]:


plot_som('h500', composites_ERA5, 'ERA5')


# In[50]:


plot_som('h500', composites_MERRA2, 'MERRA2')


# In[51]:


plot_som('h500', composites_NARR, 'NARR')


# # Correlation Analysis

# In[52]:


for node in composites_ERA5.keys():
    patt_ERA5 = composites_ERA5[node]
    patt_MERRA2 = composites_MERRA2[node]
    print('ERA5 node {} MERRA2 node {}: {}'.format(node, node, np.round(stats.pearsonr(patt_ERA5, patt_MERRA2),3)))
    


# In[53]:


for node in composites_ERA5.keys():
    patt_ERA5 = composites_ERA5[node]
    patt_NARR = composites_NARR[node]
    print('ERA5 node {} NARR node {}: {}'.format(node, node, np.round(stats.pearsonr(patt_ERA5, patt_NARR),3)))


# In[54]:


for node in composites_MERRA2.keys():
    patt_MERRA2 = composites_MERRA2[node]
    patt_NARR = composites_NARR[node]
    print('MERRA2 node {} NARR node {}: {}'.format(node, node, np.round(stats.pearsonr(patt_MERRA2, patt_NARR),3)))


# In[55]:


for node_ERA5 in composites_ERA5.keys():
    patt_ERA5 = composites_ERA5[node_ERA5]
    for node_MERRA2 in composites_MERRA2.keys():
        if node_ERA5 == node_MERRA2:
            continue
        patt_MERRA2 = composites_MERRA2[node_MERRA2]
        r, _ = stats.pearsonr(patt_ERA5, patt_MERRA2)
        if r >= 0.5:
            print('ERA5 node {} MERRA2 node {}: {}'.format(node_ERA5, node_MERRA2, stats.pearsonr(patt_ERA5, patt_MERRA2)))


# In[56]:


for node_ERA5 in composites_ERA5.keys():
    patt_ERA5 = composites_ERA5[node_ERA5]
    for node_NARR in composites_NARR.keys():
        if (node_NARR == node_ERA5):
            continue
        patt_NARR = composites_NARR[node_NARR]
        r, _ = stats.pearsonr(patt_ERA5, patt_NARR)
        if r >= 0.5:
            print('ERA5 node {} NARR node {}: {}'.format(node_ERA5, node_NARR, stats.pearsonr(patt_ERA5, patt_NARR)))


# In[57]:


for node_MERRA2 in composites_MERRA2.keys():
    patt_MERRA2 = composites_MERRA2[node_MERRA2]
    for node_NARR in composites_NARR.keys():
        patt_NARR = composites_NARR[node_NARR]
        r, _ = stats.pearsonr(patt_MERRA2, patt_NARR)
        if r >= 0.9:
            print('MERRA2 node {} NARR node {}: {}'.format(node_MERRA2, node_NARR, stats.pearsonr(patt_MERRA2, patt_NARR)))





winmap_patt = som_ERA5.win_map(data_ERA5)
#     composites = {}
#     for node in sorted(winmap_patt.keys()):
#         composites[node] = np.mean(np.array(winmap_patt[node]), axis = 0)


# In[65]    

# In[ ]:




