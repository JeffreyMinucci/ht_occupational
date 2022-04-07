#!/usr/bin/env python
# coding: utf-8

# # Bayesian modeling - version date 4/22 - for internal review

from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
import sqlite3
import pymc3 as pm
import numpy as np
import pandas as pd
import zipfile
import datetime
import os
import scipy
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
from theano import tensor as tt
from theano import shared
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
level    = logging.INFO
format   = '  %(message)s'
handlers = [logging.FileHandler('output/fit_hurdle_model.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

logging.info('Beginning fit_hurdle_model.py script.\n')

# Load data files
sector_subsector_indexes_df = pd.read_csv('output/hurdle_model_sector_subsector_domain.csv')
osha_train = pd.read_csv('output/osha_processed_training.csv')
osha_test = pd.read_csv('output/osha_processed_test.csv')
osha_full = pd.read_csv('output/osha_processed_full.csv')


# Define column names for our opera predictors and interaction terms
opera_cols = ['Intercept','logp_pred','bp_pred','loghl_pred','rt_pred','logoh_pred','logkoc_pred']
opera_cols_full = ['intercept','log octanol-water partition coefficient','boiling point', "log henry's law constant",
                 'HPLC retention time', 'log OH rate constant', 'log soil adsorption coefficient']
opera_cols_ints_names = ['logp x bp', 'logp x loghl', 'logp x rt', 'logp x logoh', 'logp x logkoc', 'bp x loghl', 'bp x rt',
                         'bp x logoh', 'bp x logkoc', 'loghl x rt', 'loghl x logoh', 'loghl x logkoc', 'rt x logoh', 'rt x logkoc', 'logoh x logkoc']


# ### Fit model part 1


sector_count = len(sector_subsector_indexes_df.index_s.unique())
sector_subsector_count = len(sector_subsector_indexes_df.index_ss.unique())

preds = osha_train[opera_cols+opera_cols_ints_names]
npreds = len(opera_cols+opera_cols_ints_names)

detected_shared = shared(osha_train.detected.values)
preds_shared = shared(preds.values)
index_s_shared = shared(osha_train.index_s.values)
index_ss_shared = shared(osha_train.index_ss.values)

# define our Bayesian hierarchical logistic model in pymc3
with pm.Model() as osha_part_1:
    
    global_mu = pm.StudentT('global_mu', nu=5, mu=0, sigma=10)
    global_lam = pm.HalfCauchy('global_lam',beta=25)
    
    sector_int_mu = pm.StudentT("sector_int_mu",nu = 5, mu = global_mu, sigma=global_lam, shape=sector_count)

    sigma_subsector_int = pm.HalfCauchy('sigma_subsector_int',beta=25)
    
    subsector_int = pm.StudentT("subsector_int", 1, mu=sector_int_mu[sector_subsector_indexes_df['index_s']],
                          sigma=sigma_subsector_int, shape=sector_subsector_count)
    beta = pm.StudentT('beta',nu=5,mu=0,sigma=2.5, shape=(npreds))  # recommended by stan wiki, Gelman et al 2008
    μ = subsector_int[index_ss_shared] + tt.dot(preds_shared,beta)
    
    θ = pm.Deterministic('θ',pm.math.sigmoid(μ))
    
    like = pm.Bernoulli(
        'likelihood',
        p = θ,
        observed=detected_shared
    )



# Fit the model using ADVI
logging.info("Fitting Bayesian hierarchical model - Part 1. May take 10+ minutes...")
with osha_part_1:
    fr_advi = pm.FullRankADVI(random_seed=2378)
tracker = pm.callbacks.Tracker(
    mean=fr_advi.approx.mean.eval,  # callable that returns mean
    std=fr_advi.approx.std.eval  # callable that returns std
)
fit1 = fr_advi.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(), tracker])
model_part_1_file = 'output/model_part1.file'
pickle.dump(fit1, open(model_part_1_file, "wb" ), pickle.HIGHEST_PROTOCOL)
logging.info("Model fitting - Part 1 completed. A copy of the part 1 model object was saved to: {}".format(model_part_1_file))


# ### Fit model part 2


osha_train_detects = osha_train[osha_train['detected'] == 1].copy()
osha_train_detects.loc[:,'log_mgm3'] = np.log10(osha_train_detects['conc_mgm3'].copy())
osha_full['log_mgm3'] = np.log10(osha_full['conc_mgm3'].copy())


# Define part 2 of the model (concentration)

preds_pt2 = osha_train_detects[opera_cols+opera_cols_ints_names].copy()
npreds = len(opera_cols+opera_cols_ints_names)

conc_shared = shared(osha_train_detects.log_mgm3.values)
preds_shared_pt2 = shared(preds_pt2.values)
index_s_shared_pt2 = shared(osha_train_detects.index_s.values)
index_ss_shared_pt2 = shared(osha_train_detects.index_ss.values)

with pm.Model() as osha_part_2:
    
    global_mu = pm.StudentT('global_mu', nu=5, mu=0, sigma=10)
    global_lam = pm.HalfCauchy('global_lam',beta=25)
    
    sector_int_mu = pm.StudentT("sector_int_mu",nu = 5, mu = global_mu, sigma=global_lam, shape=sector_count)

    sigma_subsector_int = pm.HalfCauchy('sigma_subsector_int',beta=25)
    
    subsector_int = pm.StudentT("subsector_int", 1, mu=sector_int_mu[sector_subsector_indexes_df['index_s']],
                          sigma=sigma_subsector_int, shape=sector_subsector_count)
    beta = pm.StudentT('beta',nu=5,mu=0,sigma=2.5, shape=(npreds))  # recommended by stan wiki, Gelman et al 2008
    μ = subsector_int[index_ss_shared_pt2] + tt.dot(preds_shared_pt2,beta)
    sampling_error = pm.HalfCauchy('sampling_error',beta=25)
    #θ = pm.Deterministic('θ',pm.math.sigmoid(μ))
    
    like = pm.Normal(
        'likelihood',
        mu = μ,
        sigma = sampling_error,
        observed=conc_shared
    )


#Fit model part 2 with ADVI

logging.info("Fitting Bayesian hierarchical model - Part 2. May take 10+ minutes...")
with osha_part_2:
    fr_advi = pm.FullRankADVI(random_seed=98204)
tracker = pm.callbacks.Tracker(
    mean=fr_advi.approx.mean.eval,  # callable that returns mean
    std=fr_advi.approx.std.eval  # callable that returns std
)
fit2 = fr_advi.fit(100000, callbacks=[pm.callbacks.CheckParametersConvergence(), tracker])
model_part_2_file = 'output/model_part2.file'
pickle.dump(fit2, open(model_part_2_file, "wb" ), pickle.HIGHEST_PROTOCOL)
logging.info("Model fitting - Part 1 completed. A copy of the part 2 model object was saved to: {}".format(model_part_2_file))


# ### Model coefficients plot

logging.info('Printing plot of model coefficients...')
custom_lines = [Line2D([0], [0], color='C0', lw=4),
                Line2D([0], [0], color='C1', lw=4)]
axes = az.plot_forest([fit1.sample(5000),fit2.sample(5000)], var_names=['beta'], figsize=(10,7), model_names=['Detect/Non-detect', 'Air concentration'],
                          credible_interval=0.95, linewidth = 3, markersize=10)
labels = opera_cols_full + opera_cols_ints_names
axes[0].set_yticks(np.arange(0.7,155,step=7.25))
axes[0].set_yticklabels(labels[::-1], size=14)
axes[0].set_xlabel("Scaled regression coefficient", size=16)
plt.axvline(x=0, color='black')
coefs_plot_path = 'output/fig2.png'
plt.legend(custom_lines,['Detect/Non-detect', 'Air concentration'], prop={'size': 10})
plt.savefig(coefs_plot_path)
plt.clf()
plt.close()
logging.info('Model coefficient plot saved to: {}'.format(coefs_plot_path))


# ### Training set performance

logging.info('Calculating training set model performance...')
# Training set performance - part 1
np.random.seed(3945)
ppc_train = pm.sample_posterior_predictive(fit1.sample(5000), samples= 1000, random_seed=44749, model=osha_part_1)
avg_predictions_train = np.where(ppc_train['likelihood'].mean(axis=0) >= 0.5, 1, 0)
prob_predictions_train = ppc_train['likelihood'].mean(axis=0)
accuracy_train = accuracy_score(osha_train['detected'], avg_predictions_train)
auc_train = roc_auc_score(osha_train['detected'], prob_predictions_train)
logging.info('Part 1 - Training classification accuracy: {:.3f}'.format(accuracy_train))
logging.info('Part 1 - Training ROC area under the curve: {:.3f}'.format(auc_train))

np.random.seed(54296)
null_predictions_train = np.where(np.random.uniform(size=len(osha_train['detected'])) >
                                        osha_train['detected'].mean(),1,0)
null_probs_train = np.full(shape=len(osha_train['detected']), fill_value=osha_train['detected'].mean())
null_accuracy = accuracy_score(null_predictions_train,osha_train['detected'])
null_auc = roc_auc_score(osha_train['detected'],null_probs_train)
logging.info('Part 1 - Null model training classification accuracy: {:.3f}'.format(null_accuracy))
logging.info('Part 1 - Null model training ROC area under the curve: {:.3f}'.format(null_auc))



# Training set performance - part 2
ppc_pt2_train = pm.sample_posterior_predictive(fit2.sample(5000), samples= 1000, random_seed=6584, model=osha_part_2)
avg_predictions_pt2_train = ppc_pt2_train['likelihood'].mean(axis=0)
rmse_train = np.sqrt(mean_squared_error(osha_train_detects['log_mgm3'], avg_predictions_pt2_train))

np.random.seed(25855)
null_predictions_pt2_train = np.repeat(osha_train_detects['log_mgm3'].mean(),
                                      repeats=len(osha_train_detects['log_mgm3']))
null_rmse = np.sqrt(mean_squared_error(osha_train_detects['log_mgm3'], null_predictions_pt2_train))
logging.info('Part 2 - Training RMSE: {:.2f}'.format(rmse_train))
logging.info('Part 2 - Null model training RMSE: {:.2f}'.format(null_rmse))

# Generate pt 2 predictions for all training data, including non-detects
preds_full_conc = osha_train[opera_cols+opera_cols_ints_names]
full_train_dummy_conc = np.log10(osha_train['conc_mgm3'])
#full_train_dummy_conc[~np.isfinite(full_train_dummy_conc)] = -9999
conc_shared.set_value(full_train_dummy_conc.values)
preds_shared_pt2.set_value(preds_full_conc.values)
index_s_shared_pt2.set_value(osha_train.index_s.values)
index_ss_shared_pt2.set_value(osha_train.index_ss.values)
ppc_full_train_pt2 = pm.sample_posterior_predictive(fit2.sample(5000), samples=1000, random_seed=6596, model=osha_part_2) # rows = 500 model predictions, cols = 45800 osha samples
avg_predictions_pt2_full_train = ppc_full_train_pt2['likelihood'].mean(axis=0)



# ### Test set performance

logging.info('Calculating test set model performance...')
# Test set performance - part 1
preds_test = osha_test[opera_cols+opera_cols_ints_names]
detected_shared.set_value(osha_test.detected.values)
preds_shared.set_value(preds_test.values)
index_s_shared.set_value(osha_test.index_s.values)
index_ss_shared.set_value(osha_test.index_ss.values)
np.random.seed(18373)
sam = fit1.sample(5000)
ppc_test = pm.sample_posterior_predictive(sam, samples= 1000, random_seed=13464, model=osha_part_1)
avg_predictions_test = np.where(ppc_test['likelihood'].mean(axis=0) >= 0.5, 1, 0)
avg_probs_test = ppc_test['likelihood'].mean(axis=0)

np.random.seed(986865)
test_accuracy = accuracy_score(avg_predictions_test, osha_test['detected'])
null_predictions_test = np.where(np.random.uniform(size=len(osha_test['detected'])) >  # using null model 'trained' on training set
                                        osha_test['detected'].mean(),1,0)
null_probs_test = np.full(shape=len(osha_test['detected']), fill_value=osha_train['detected'].mean())
null_accuracy = accuracy_score(osha_test['detected'], null_predictions_test)
test_auc = roc_auc_score(osha_test['detected'], avg_probs_test)
null_auc = roc_auc_score(osha_test['detected'],null_probs_test)
logging.info('Part 1 - Test classification accuracy: {:.3f}'.format(test_accuracy))
logging.info('Part 1 - Test ROC area under the curve: {:.3f}'.format(test_auc))
logging.info('Part 1 - Null model test classification accuracy: {:.3f}'.format(null_accuracy))
logging.info('Part 1 - Null model test ROC area under the curve: {:.3f}'.format(null_auc))


# Test set performance - part 2

osha_test_c = osha_test.copy()
osha_test_c['pred_detected'] = avg_predictions_test
osha_test_c['log_mgm3'] = np.log10(osha_test_c['conc_mgm3'])
preds_pt2_test = osha_test_c[opera_cols + opera_cols_ints_names]
conc_shared.set_value(osha_test_c.log_mgm3.values)
preds_shared_pt2.set_value(preds_pt2_test.values)
index_s_shared_pt2.set_value(osha_test_c.index_s.values)
index_ss_shared_pt2.set_value(osha_test_c.index_ss.values)

ppc_p2_test = pm.sample_posterior_predictive(fit2.sample(5000), samples= 1000, random_seed=23094, model=osha_part_2)
avg_predictions_pt2_test = ppc_p2_test['likelihood'].mean(axis=0)

test_combined = osha_test_c.copy()
test_combined['pred_log_mgm3'] =  avg_predictions_pt2_test
test_combined['pred_mgm3'] = 10 ** avg_predictions_pt2_test

actual = test_combined['log_mgm3'][(test_combined['pred_detected'] == 1) & (test_combined['detected'] == 1)]
predicted = avg_predictions_pt2_test[(test_combined['pred_detected'] == 1) & (test_combined['detected'] == 1)]
rmse_test_TPs = np.sqrt(mean_squared_error(actual, predicted))
logging.info('Part 2 - Test RMSE for true positives: {:.2f}'.format(rmse_test_TPs))

np.random.seed(11231)
test_combined['null_pred_detected'] = np.where(np.random.uniform(size=len(test_combined['detected'])) >  # using null model 'trained' on training set
                                        osha_train['detected'].mean(),1,0)
train_mean_log_mgm3 = np.log10(osha_train[osha_train.detected == 1].conc_mgm3).mean()
test_combined['null_pred_log_mgm3'] = np.repeat(train_mean_log_mgm3,
                                      repeats=len(test_combined['log_mgm3']))

actual = test_combined['log_mgm3'][(test_combined['null_pred_detected'] == 1) & (test_combined['detected'] == 1)]
null_predicted = test_combined['null_pred_log_mgm3'][(test_combined['null_pred_detected'] == 1) & (test_combined['detected'] == 1)]
test_mean_rmse = np.sqrt(mean_squared_error(actual,null_predicted))
logging.info('Part 2 - Null model test RMSE for true positives: {:.2f}'.format(test_mean_rmse))


# Plot test set perfomance for concentration

# Plot actual vs predicted for actual positives
actual_tp = test_combined['log_mgm3'][(test_combined['pred_detected'] == 1) & (test_combined['detected'] == 1)]
predicted_tp = avg_predictions_pt2_test[(test_combined['pred_detected'] == 1) & (test_combined['detected'] == 1)]
actual_fn = test_combined['log_mgm3'][(test_combined['pred_detected'] == 0) & (test_combined['detected'] == 1)]
predicted_fn = avg_predictions_pt2_test[(test_combined['pred_detected'] == 0) & (test_combined['detected'] == 1)]
plt.scatter(predicted_fn, actual_fn, s=10, color="red", alpha=1, label='Predicted non-detect')
plt.scatter(predicted_tp, actual_tp, s=10, alpha=1, label='Predicted detect')
plt.plot([-20, 10], [-20, 10], color='black')
plt.xlim([-5,6])
plt.ylim([-5,6])
plt.title("Test set")
plt.xlabel('Predicted air concentration (log mg/m3)')
plt.ylabel('Actual air concentration (log mg/m3)')
plt.text(-4.5, 2.5, 'RMSE (true pos) = {:.2f}'.format(rmse_test_TPs))
plt.legend()
plot_path = 'output/test_set_concentration_plot.png'
plt.savefig(plot_path)
logging.info('Plot of test set actual vs predicted concentration saved to: {}'.format(plot_path))



# Save OSHA model predictions for training and test dataset
output_train = osha_train.copy()
output_train['log_mgm3'] =  np.log10(output_train['conc_mgm3'])
output_train.loc[~np.isfinite(output_train['log_mgm3']),'log_mgm3'] = 'ND'
output_train['pred_detected'] = avg_predictions_train
output_train['pred_log_mgm3'] = avg_predictions_pt2_full_train
output_train['pred_log_mgm3_final'] = np.where(output_train['pred_detected']==1,output_train['pred_log_mgm3'],'ND')
train_out_path = 'output/osha_train_predictions.csv'
output_train.to_csv(train_out_path, index=False)
logging.info('OSHA training set with model predictions saved to: {}'.format(train_out_path))

output_test = osha_test.copy()
output_test['log_mgm3'] =  np.log10(output_test['conc_mgm3'])
output_test.loc[~np.isfinite(output_test['log_mgm3']),'log_mgm3'] = 'ND'
output_test['pred_detected'] = avg_predictions_test
output_test['pred_log_mgm3'] = avg_predictions_pt2_test
output_test['pred_log_mgm3_final'] = np.where(output_test['pred_detected']==1,output_test['pred_log_mgm3'],'ND')
test_out_path = 'output/osha_test_predictions.csv'
output_test.to_csv(test_out_path, index=False)
logging.info('OSHA test set with model predictions saved to: {}'.format(test_out_path))

logging.info('Completed fit_hurdle_model.py script.\n')
