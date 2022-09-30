import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Import data from CSV.

df = pd.read_csv('landmarks.csv', index_col = 0)



# Define functions for computing geometric measures of symmetry ('symmetrics').

symmetrics = ['osa', 'rfs', 'fa', 'ga', 'td', 'hhd']

def angle(v1, v2, radians = False): # Signed angle.
    v1, v2 = np.array(v1), np.array(v2)
    angle = math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)
    return angle if radians else 180 * angle / np.pi

def extract_tort_coords(row, model):
    selected_col_names = [model + '-' + str(ax) + str(lmk) for ax in ['x', 'y'] for lmk in range(70)] 
    selected_cols = [col for col in selected_col_names]
    return row[selected_cols].to_numpy(dtype = np.float64).reshape((2, 70))

def compute_symmetrics(row, symmetric, model, radians = False):
    assert symmetric in symmetrics
    
    coords = np.array(extract_tort_coords(row, model))
    result = None
    
    # Start with general quantities used in multiple symmetrics.
    outer_slope = coords[ : , 36] - coords[ : , 45]
    shoulder_line = coords[ : , 68] - coords [ : , 69]
    
    # Define eyeline relative to corners of eyes, rather than all
    # eye coordinates, so that it is valid even when the eyes are 
    # closed (and the inner eye coordinates all lie below the eyeline).
    left_eye_mid = (coords[ : , 42] + coords[ : , 45]) / 2
    right_eye_mid = (coords[ : , 36] + coords[ : , 39]) / 2
    eye_line = right_eye_mid - left_eye_mid
    
    if symmetric == 'osa':
        inner_slope = coords[ : , 39] - coords[ : , 42]
        result = angle(outer_slope, inner_slope, radians)
    elif symmetric == 'rfs':
        left_canthus = coords[ : , 45] - coords[ : , 54]
        right_canthus = coords[ : , 36] - coords[ : , 48]
        left_norm = (left_canthus @ left_canthus) ** (1/2)
        right_norm = (right_canthus @ right_canthus) ** (1/2)
        result = left_norm / right_norm
        # result = max(left_norm / right_norm, right_norm / left_norm)
    elif symmetric == 'fa':
        lip_line = coords[ : , 48] - coords[ : , 54]
        result = angle(eye_line, lip_line, radians)
    elif symmetric == 'ga':
        shoulder_line_perp = - np.array([shoulder_line[1], - shoulder_line[0]])
        result = angle(outer_slope, shoulder_line_perp) 
    elif symmetric == 'td':
        shoulder_midpt = (coords[ : , 68] + coords [ : , 69]) / 2
        chin = coords[ : , 8]
        transl_def = ((chin - shoulder_midpt) @ shoulder_line) / \
                      ((shoulder_line @ shoulder_line) ** (1 / 2)) 
                         # Projection of (chin - shoulder_midpt) onto shoulder_line
        norm_trans_def = transl_def / ((outer_slope @ outer_slope) ** (1/2))
                             # Normalize by outer slope length
        result = abs(norm_trans_def)
    elif symmetric == 'hhd':
        result = angle(eye_line, shoulder_line, radians)
        
    return result



# Create analysis dataframe extending the coordinate dataframe with
# computed symmetrics.

radians = False
tort_models = ['gt', 'infant', 'adult']
   
# Compute errors for each file and store in a numpy array. 
symmetric_array = np.empty((len(df.index), len(tort_models) * len(symmetrics)))    # Rows correspond to images

for row_index, row in df.iterrows():
    symmetric_list = []    # Will contain all symmetrics for the current image
    for model in tort_models: 
        for symmetric in symmetrics:
            symmetric_list.append(compute_symmetrics(row, symmetric, model, radians))
    symmetric_array[row_index] = np.array(symmetric_list)

# Neatly repackage the data into a dataframe.
symmetrics_df = pd.DataFrame(symmetric_array)
symmetrics_df.columns = [model + '-' + symmetric for model in tort_models 
                                                 for symmetric in symmetrics]
full_symmetrics_df = df.join(symmetrics_df)

# Drop landmarks and attributes.
landmark_cols = [mod + '-' + str(ax) + str(lmk) for mod in tort_models \
                 for ax in ['x', 'y'] for lmk in range(70)]
symmetrics_df = full_symmetrics_df.drop(landmark_cols, axis = 1)



# Compute performance metrics.

mae = np.zeros((2, 6))
rmse = np.zeros((2, 6))
spearman = np.zeros((2, 6))
spearmanp = np.zeros((2, 6))
mean = np.zeros((2, 6))
std = np.zeros((2, 6))
acc = np.zeros((2, 6))

for m, mod in enumerate(['infant', 'adult']):
    for i, sym in enumerate(['osa', 'rfs', 'fa', 'ga', 'td', 'hhd']):
        mae[m, i] = (symmetrics_df['gt-' + sym] - symmetrics_df[mod + '-' + sym]).abs().mean()
        rmse[m, i] = ((symmetrics_df['gt-' + sym] - symmetrics_df[mod + '-' + sym]) ** 2).mean() ** (1/2)
        
        spearman_pair = spearmanr(symmetrics_df['gt-' + sym], symmetrics_df[mod + '-' + sym])
        spearman[m, i] = spearman_pair[0]
        spearmanp[m, i] = spearman_pair[1]
        
        sym_min, sym_max = symmetrics_df['gt-' + sym].min(), symmetrics_df['gt-' + sym].max()
        mean[m, i] = symmetrics_df['gt-' + sym].mean()
        std[m, i] = symmetrics_df['gt-' + sym].std()
        acc[m, i] = sum((symmetrics_df['gt-' + sym] > mean[m, i]) \
                        == (symmetrics_df[mod + '-' + sym] > mean[m, i]))/36
        
        print('geometric measure:', sym)
        print('pose estimation model:', mod)
        print('mae\t\t', mae[m, i])
        print('rmse\t\t', rmse[m, i])
        print('spearman\t', spearman[m, i])
        print('spearmanp\t', spearmanp[m, i])
        print('mean\t\t', mean[m, i])
        print('std\t\t', std[m, i])
        print('acc\t\t', acc[m, i])
        print('\n')
        
        
        
# Scatter plots of ground truth vs predictions.

fig, ax = plt.subplots(2, 6, figsize = (14, 7), dpi = 200)

y_lims = [[-80, 20], [-0.05, 1.4], [-70, 15], [-20, 110], [-0.05, 0.5], [-80, 20]]

for m, mod in enumerate(['infant', 'adult']):
    for i, sym in enumerate(['osa', 'rfs', 'fa', 'ga', 'td', 'hhd']):
        sym_min, sym_max = symmetrics_df['gt-' + sym].min(), symmetrics_df['gt-' + sym].max()
        title = sym + '\nmae: ' + '{0:.2f}'.format(mae[m, i]) \
                    + '\nrmse: ' + '{0:.2f}'.format(rmse[m, i]) \
                    + '\nspearman rho: ' + '{0:.2f}'.format(spearman[m, i]) \
                    + '\nspearman rho p: ' + '{0:.5f}'.format(spearmanp[m, i]) \
                    + '\nmean: ' + '{0:.2f}'.format(mean[m, i]) \
                    + '\nacc: ' + '{0:.3f}'.format(acc[m, i])
        symmetrics_df.plot.scatter(ax = ax[m, i], x = 'gt-' + sym, y = mod + '-' + sym, c = 'r')
        ax[m, i].plot([sym_min, sym_max], [sym_min, sym_max], c = 'r')
        ax[m, i].set_ylim(y_lims[i])
        ax[m, i].set_title(title)
        
        if m == 0:
            ax[m, i].set_xlabel('')
        elif m == 1:
            ax[m, i].set_xlabel('Ground Truth')
            
        if i == 0:
            if m == 0:
                ax[m, i].set_ylabel('Infant Model\nPrediction')
            else:
                ax[m, i].set_ylabel('Adult Model\nPrediction')
        else:
            ax[m, i].set_ylabel('')

fig.tight_layout()

plt.savefig('scatter.jpg', dpi = 'figure')