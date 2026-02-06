# In[1]:
import ipywidgets as widgets
from IPython.display import display

import random 
import time
from pathlib import Path
import numpy as np
import pandas as pd

import autoeis as ae
from funcs import *

# Set this to True if you're running the notebook locally
# interactive = True
# ae.visualization.set_plot_style()


# In[2]:

## USER INPUT ####
method = 'log-BW'    # Methods are:  "UW"  "X2"  "PW"  "B"  "log-B"   "log-BW"
fit_method = 'local' # 'local' or 'global' or 'local+global'
##################


# In[3]:  Load the data
dataset = pd.read_pickle('../target_circuits.pkl')
dataset['z_mag_diff'] = dataset['Z_true'].apply(lambda x: np.max(np.abs(x))-np.min(np.abs(x)))
dataset

# In[4]:

def fit_circuit_eval(circuit_string, freq, Z_true, loss_function, fit_method = 'local'):
        """
        fit_method = 'local' use only the local optimizer
        fit_method = 'global' use only the global optimizer
        fit_method = 'local+global' use both local and global optimizers -> that is it first use local to find a good starting point and then use global to find the best solution
        """
        initial_time = time.time()
        param = None
        if fit_method == 'local' or fit_method == 'local+global':
                param, X2_internal, r2_mag, r2_phase, perror, converged = fit_circuit_parameters_NEW(circuit_string, freq, Z_true, min_iters = 5, max_iters=20, method= loss_function, tol_chi_squared=1e-2)
                
        if fit_method == 'global'or fit_method == 'local+global':
                param, X2_internal, r2_mag, r2_phase, perror, converged = fit_circuit_global_min(circuit_string, freq, Z_true, method= loss_function, p0 = param)    # for loss_function choose from "UW"  "X2"  "PW"  "B"  "log-B"   "log-BW"
                
        required_time = time.time() - initial_time
        param_list = [values for key, values in param.items()]
        number_param = len(param_list)
        np_pram = np.array(param_list)
        predicted_Z = ae.utils.eval_circuit(circuit_string, freq, np_pram)
        chi_square = chi_obj_func(Z_true, predicted_Z, number_param)
        r2_score = ae.metrics.r2_score(Z_true, predicted_Z)
        # print("circuit:", circuit)
        # print("r2_score:", r2_score)
        # print("chi_square:", chi_square)
        output = {'r2_score': r2_score, 'chi_square': chi_square, 'X2_internal': X2_internal ,'r2_mag': r2_mag, 'r2_phase': r2_phase,  "converged": converged, 'time': required_time, 'predicted_Z': predicted_Z, 'error': perror, 'param': param, 'param_list': param_list}
        return output

# In[5]:

dir = f'loss_func_{method}'
dir_path = Path(dir)
dir_path.mkdir(exist_ok=True)
plot_path = Path(f'{dir}/plots')
plot_path.mkdir(exist_ok=True)
plot_fits = False

random.seed(42)
np.random.seed(42)

metrics = pd.DataFrame(columns=['dataset','circuit_string', 'r2_score', 'chi_square', 'X2_internal', 'r2_mag', 'r2_phase', 'converged', 'time', 'error', 'true_param', 'param', 'Z_true'])
for row in dataset.iterrows():
    # print("Working on EIS number: ",row[0])
    # state = row[1]['state']
    circuit_string = row[1]['circuit_string']
    freq = np.array(row[1]['freq'])
    Z_true = np.array(row[1]['Z_true'])
    true_param = row[1]['component_values']

    output = fit_circuit_eval(circuit_string, freq, Z_true, loss_function=method, fit_method=fit_method) 
    r2_score = output['r2_score']
    chi_square = output['chi_square']
    X2_internal = output['X2_internal']
    r2_mag = output['r2_mag']
    r2_phase = output['r2_phase']
    converged = output['converged']
    error = output['error']
    required_time = output['time']
    predicted_Z = output['predicted_Z']
    param = output['param']
    param_list = output['param_list']

    metrics = pd.concat([metrics, pd.DataFrame({'dataset': [row[0]], 'circuit_string': [circuit_string], 'r2_score': [r2_score], 'chi_square': [chi_square], 'X2_internal': [X2_internal],'r2_mag': [r2_mag], 'r2_phase': [r2_phase], 'converged': [int(converged)], 'time': [required_time], 'error': [error], 'true_param': [true_param], 'param': [param], 'Z_true': [Z_true], })], ignore_index=True)
    # print('r2_score:', r2_score, 'chi_square:', chi_square, 'X2_internal:', X2_internal,'r2_mag:', r2_mag, 'r2_phase:', r2_phase, 'converged:', converged)
    if plot_fits:
        plot_nyquist_bode(predicted_Z, Z_true, freq, titile_nyquist=f'nyquist_{row[0]}', title_bode_phase=f'bode_phase_{row[0]}', title_bode_mag = f'bode_mag_{row[0]}', show = False, save=True, name = f'{row[0]}_{circuit_string}', dir= plot_path, convergance= converged)
    

metrics.to_pickle(f'metrics_{method}.pkl')
metrics.to_csv(f'metrics_{method}.csv')
print("Done")




# %%
