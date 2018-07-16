# Predict future sale assignment
# import libraries

import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.sparse 

from sklearn.metrics import mean_squared_error,make_scorer
from math import sqrt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb

import os
import gc
import pickle
import progressbar
widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]

data_path = "/Users/juanzinser/Workspace/advance-machine-learning/cds/competition/data/"
pbar = progressbar.ProgressBar(widgets=widgets)

for p in [np, pd, scipy, sklearn]:
    print (p.__name__, p.__version__)

sales    = pd.read_csv(data_path + 'sales_train_v2.csv')
items           = pd.read_csv(data_path + 'items.csv',encoding ='ISO-8859-1')
item_categories = pd.read_csv(data_path + 'item_categories.csv',encoding ='ISO-8859-1')
shops           = pd.read_csv(data_path + 'shops.csv',encoding ='ISO-8859-1')
test            = pd.read_csv(data_path + 'test.csv')

# Utilities used in the assignment
def downcast_dtypes(df):
	'''
	Changes column types in the dataframe: 
		
		`float64` type to `float32`
		`int64`   type to `int32`
	'''

	# Select columns to downcast
	float_cols = [c for c in df if df[c].dtype == "float64"]
	int_cols =   [c for c in df if df[c].dtype == "int64"]

	# Downcast
	df[float_cols] = df[float_cols].astype(np.float32)
	df[int_cols]   = df[int_cols].astype(np.int32)

	return df

def root_mean_squared_error(truth,pred):
	return sqrt(mean_squared_error(truth,pred))

def get_all_data(filename):
	all_data = pd.read_pickle(filename)
	all_data = downcast_dtypes(all_data)
	all_data = all_data.reset_index().drop('index',axis=1)
	return all_data

def get_cv_idxs(df,start,end):
	result=[]
	for i in range(start,end+1):
		dates = df.date_block_num
		train_idx = np.array(df.loc[dates <i].index)
		val_idx = np.array(df.loc[dates == i].index)
		result.append((train_idx,val_idx))
	return np.array(result)
def get_X_y(df,end,clip=20):
	# don't drop date_block_num
	df = df.loc[df.date_block_num <= end]
	cols_to_drop=['target','item_name'] + df.columns.values[6:12].tolist()
	y = np.clip(df.target.values,0,clip)
	X = df.drop(cols_to_drop,axis=1)
	return X,y


sales.groupby('date_block_num').agg({"date_block_num":"count"}).plot(figsize=(10,6))

sales["shop_id"].nunique(), sales["item_id"].nunique()
test["shop_id"].nunique(), test["item_id"].nunique()

# Attach date block num column 34 in test data. Prediction month
test['date_block_num'] = 34
test.drop('ID', axis = 1, inplace =True)
temp_test = test.copy()

# Feature engineering
# Remove outliers
sales = sales[sales.item_cnt_day<=1000]  # Only one value
sales = sales[sales.item_price<100000]

# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

# Fix column names
print(gb.columns.values)
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]

gb.head()

# Join sales data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

# merge with category id
all_data =pd.merge(all_data,items,on=['item_id'],how='left')
# include item category id in test set
temp_test = pd.merge(temp_test, items, on=['item_id'], how = 'left')

# Same as above but with shop-month aggregates
gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'shop_block_target_sum':'sum','shop_block_target_mean':np.mean}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

all_data = downcast_dtypes(all_data)

# Same as above but with item-month aggregates
gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'item_block_target_sum':'sum','item_block_target_mean':np.mean}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# Same as above but with item category-month aggregates
sales = pd.merge(sales,items,on=['item_id'],how='left')
gb = sales.groupby(['item_category_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'item_cat_block_target_sum':'sum','item_cat_block_target_mean':np.mean}})
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)

del grid, gb 
gc.collect();

all_data.shape

all_data_merge = pd.concat([all_data, temp_test], axis = 0, ignore_index = True)
# Align the columns given initially
all_data = all_data_merge[['shop_id', 'item_id', 'date_block_num', 'target', 'item_name',
                           'item_category_id', 'shop_block_target_sum', 'shop_block_target_mean',
                           'item_block_target_sum', 'item_block_target_mean',
                           'item_cat_block_target_sum', 'item_cat_block_target_mean']]

all_data = downcast_dtypes(all_data)
del all_data_merge
index_cols = ['shop_id', 'item_id', 'date_block_num','item_category_id']
cols_to_rename = list(all_data.columns.difference(index_cols))
for i in ['item_name']:
    cols_to_rename.remove(i)
print(cols_to_rename)
cols_gb_item = [i for i in cols_to_rename if 'item_block' in i]
cols_gb_shop = [i for i in cols_to_rename if 'shop_block' in i]
cols_gb_cat = [i for i in cols_to_rename if 'item_cat' in i]
cols_gb_all = ['target']
cols_gb_key=[['item_id'],['shop_id'],['item_category_id'],['shop_id','item_id']]
cols_gb_value = [cols_gb_item,cols_gb_shop,cols_gb_cat,cols_gb_all]
print(cols_gb_value)


shift_range = [1,2,3,5,12]
for month_shift in pbar(shift_range):
    for k,v in zip(cols_gb_key,cols_gb_value): 
        index_col = ['date_block_num'] + k
        train_shift = all_data[index_col + v].copy().drop_duplicates()

        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in v else x
        train_shift = train_shift.rename(columns=foo)
        all_data = pd.merge(all_data, train_shift, on=index_col, how='left').fillna(0)
        
all_data.shape

# Add Holidays features
all_data['December'] = all_data.date_block_num.apply(lambda x: 1 if x == 23  else 0)

all_data['Newyear_Christmas'] = all_data.date_block_num.apply(lambda x: 1 if x in [12,24] else 0)
all_data['Valentine_MenDay'] = all_data.date_block_num.apply(lambda x: 1 if x in [13,25] else 0)
all_data['WomenDay'] = all_data.date_block_num.apply(lambda x: 1 if x in [14,26] else 0)
all_data['Easter_LaborDay'] = all_data.date_block_num.apply(lambda x: 1 if x in [15,27] else 0)
 
all_data.to_pickle('new_sales_train_test_lag_data.pickle')

train_test_lag = get_all_data('new_sales_train_test_lag_data.pickle')

all_data_train = train_test_lag[train_test_lag.date_block_num <= 33]
test_lag = train_test_lag[train_test_lag.date_block_num == 34]

X,y = get_X_y(all_data_train,33)
X.drop('date_block_num',axis=1,inplace=True)
cv = get_cv_idxs(all_data_train,28,33)

cols_to_drop=['target','item_name','date_block_num'] + test_lag.columns.values[6:12].tolist()
X_test = test_lag.drop(cols_to_drop,axis=1)

# Create light GBM model 

cv_loss_list=[]
n_iteration_list=[]
def score(params):
    print("Training with params: ")
    print(params)
    cv_losses=[]
    cv_iteration=[]
    for (train_idx,val_idx) in cv:
        print("train idx", train_idx)
        print("val idx", val_idx)
        cv_train = X.iloc[train_idx]
        cv_val = X.iloc[val_idx]
        cv_y_train = y[train_idx]
        cv_y_val = y[val_idx]
        lgb_model = lgb.train(params, lgb.Dataset(cv_train, label=cv_y_train), 2000, 
                          lgb.Dataset(cv_val, label=cv_y_val), verbose_eval=False, 
                          early_stopping_rounds=100)
       
        train_pred = lgb_model.predict(cv_train,lgb_model.best_iteration+1)
        val_pred = lgb_model.predict(cv_val,lgb_model.best_iteration+1)
        
        val_loss = root_mean_squared_error(cv_y_val,val_pred)
        train_loss = root_mean_squared_error(cv_y_train,train_pred)
        print('Train RMSE: {}. Val RMSE: {}'.format(train_loss,val_loss))
        print('Best iteration: {}'.format(lgb_model.best_iteration))
        cv_losses.append(val_loss)
        cv_iteration.append(lgb_model.best_iteration)
    print('6 fold results: {}'.format(cv_losses))
    cv_loss_list.append(cv_losses)
    n_iteration_list.append(cv_iteration)
    
    mean_cv_loss = np.mean(cv_losses)
    print('Average iterations: {}'.format(np.mean(cv_iteration)))
    print("Mean Cross Validation RMSE: {}\n".format(mean_cv_loss))
    return {'loss': mean_cv_loss, 'status': STATUS_OK}

def optimize(space,seed = seed, max_evals = 5):
    
    best = fmin(score, space, algo=tpe.suggest, 
        # trials=trials, 
        max_evals=max_evals)
    return best

space = {
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'min_data_in_leaf': hp.choice('min_data_in_leaf',np.arange(5, 30,1, dtype=int)),
    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'seed':seed,
    'objective': 'regression',
    'metric':'rmse',
}
best_hyperparams = optimize(space,5)
print("The best hyperparameters are: ")
print(best_hyperparams)

# Start prediction for the hold out test set
lgb_params = {
               'colsample_bytree': 0.75,
               'metric': 'rmse',  
               'min_data_in_leaf': 128, 
               'subsample': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'regression', 
               'bagging_seed': 128, 
               'num_leaves': 128,
               'bagging_freq':1,
               'seed':1204
              }

gc.collect()

lgb_model_full = lgb.train(lgb_params, lgb.Dataset(X, label=y), 708, 
                      lgb.Dataset(X, label=y), verbose_eval=10)

pickle.dump(lgb_model_full, open("lgb.pickle.dat", "wb"))

lgb_model_full = pickle.load(open("lgb.pickle.dat", "rb"))

test_pred = lgb_model_full.predict(X_test,708)

test_sub   = pd.read_csv('test.csv')
test_sub.drop(['shop_id', 'item_id', 'ID'], axis = 1, inplace = True)
test_sub['item_cnt_month'] = test_pred

test_sub.to_csv('test_submission.csv')
