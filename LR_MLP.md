# Model Training and Evalaution

**Define Parameters for the model**
```python
RF_tuned_params = {'max_depth': 60, 
                   'max_features': 50, 
                   'min_samples_leaf': 1,
                   'n_estimators': 500,  
                   'n_jobs': -1, 
                   'random_state': 42
                  }
RF = RandomForestClassifier().set_params(**RF_tuned_params)

LR_tuned_params = {'C': 1, 
                   'penalty': 'l1', 
                   'solver': 'liblinear', 
                   'random_state': 42
                  }
LR = LogisticRegression().set_params(**LR_tuned_params)

MLP_tuned_params = {'hidden_layer_sizes': (55), 
                    'activation': 'relu', 
                    'solver': 'adam', 
                    'max_iter': 1000 
                   }
MLP = MLPClassifier(**MLP_tuned_params)
```

**Initialize array for Residue Importance**

Apo and mmv will hencforth represent the unbound(0) and bound(1) states

```python
def sum_elements(i_array):
    resid = []
    for i in i_array:
        if i[0] not in resid:
            resid.append(i[0]) 
    import_sum = []
    for i in resid:
        su = 0
        for j in i_array:
            if j[0] == i:
                su = su + j[1]
        import_sum.append([i,su])
    import_sum = np.array(import_sum)
    xmax, ymax = import_sum.max(axis=0)
    import_sum[:,1] /= ymax
    import_sum = import_sum[import_sum[:,0].argsort()]
    return import_sum
```

**LR training and testing**
```python
f1, prec, recall, acc, ROC_AUC, conf = ([], [], [], [], [], [])
for run in range(0, 20):
    X = df_dropped.iloc[:, :-1]    
    y = df_dropped['State']
    X, y = X.to_numpy(), y.to_numpy()
    df_lr_coef = pd.DataFrame(columns=df_dropped.columns[:-1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    y_test_compare = y_test
    for rndm_state in range(0, 10):
        X_train_resampl, y_train_resampl = resample(
            X_train, y_train,
            n_samples=len(X_train),
            random_state=rndm_state
        )
        y_train_resampl = y_train_resampl.astype(int)
        LR.fit(X_train_resampl, y_train_resampl.ravel())
        y_pred = LR.predict(X_test)
        y_test_compare = y_test_compare.astype(int)
        y_pred = y_pred.astype(int)
        acc.append(accuracy_score(y_test_compare, y_pred))
        f1.append(f1_score(y_test_compare, y_pred))
        prec.append(precision_score(y_test_compare, y_pred))
        recall.append(recall_score(y_test_compare, y_pred))
        ROC_AUC.append(roc_auc_score(y_test_compare, y_pred))
        new_data = pd.DataFrame(LR.coef_, columns=df_dropped.columns[:-1])
        df_lr_coef = pd.concat([df_lr_coef, new_data], ignore_index=True)
    df_lr_coef_mean_abs = abs(df_lr_coef.mean().to_frame().T)
    apo_impor_resids = []
    mmv_impor_resids = []
    for col in df_lr_coef_mean_abs.columns.values:
        importance = df_lr_coef_mean_abs.at[0, col]
        r1, r2 = extract_residues(col)
        if r1 is None:
            continue
        apo_impor_resids.append([r1, importance])
        apo_impor_resids.append([r2, importance])
        mmv_impor_resids.append([r1, importance])
        mmv_impor_resids.append([r2, importance])
    lr_apo_impor_resids_sum = sum_elements(apo_impor_resids)
    lr_mmv_impor_resids_sum = sum_elements(mmv_impor_resids)
    LR_impo_res_apo = pd.concat([
        LR_impo_res_apo,
        pd.DataFrame(
            np.reshape(lr_apo_impor_resids_sum[:, 1], (1, len(lr_apo_impor_resids_sum))),
            columns=lr_apo_impor_resids_sum[:, 0]
        )
    ], ignore_index=True)

    LR_impo_res_mmv = pd.concat([
        LR_impo_res_mmv,
        pd.DataFrame(
            np.reshape(lr_mmv_impor_resids_sum[:, 1], (1, len(lr_mmv_impor_resids_sum))),
            columns=lr_mmv_impor_resids_sum[:, 0]
        )
    ], ignore_index=True)
LR_impo_res_apo = LR_impo_res_apo.fillna(0)
LR_impo_res_mmv = LR_impo_res_mmv.fillna(0)
#Plot accuracy calculated from training the earlier model
x = np.linspace(0, len(acc), len(acc))
fig = plt.subplots(figsize=(8,2))
plt.scatter(x,acc, color='k', s=20)
plt.ylim(0,1.1)
```

**RF Training and Testing**
```python
f1,prec,recall,acc,ROC_AUC,conf = ([],[],[],[],[],[])
for run in range(0,20):
    X = df_dropped.iloc[:,:-1]
    y = df_dropped['State']
    X, y = X.to_numpy(),y.to_numpy()
    num_features = X.shape[1]
    df_rf_coef = pd.DataFrame(columns = df_dropped.columns[:-1])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 42) #same random_state as previous GS
    for rndm_state in range(0,1) :
        X_train_resampl,y_train_resampl = resample(X_train,y_train, n_samples=len(X_train), random_state=rndm_state)
        y_train_resampl=y_train_resampl.astype('int')
        RF.fit(X_train_resampl, y_train_resampl.ravel())
        y_pred = RF.predict(X_test)
        y_test_compare = y_test_compare.astype(int)
        y_pred = y_pred.astype(int)
        acc.append(accuracy_score(y_test_compare, y_pred))
        f1.append(f1_score(y_test_compare, y_pred))
        prec.append(precision_score(y_test_compare, y_pred))
        recall.append(recall_score(y_test_compare, y_pred))
        ROC_AUC.append(roc_auc_score(y_test_compare, y_pred))
        new_data = pd.DataFrame(np.reshape(RF.feature_importances_, (1,num_features)), columns = df_dropped.columns[:-1])
        df_rf_coef = pd.concat([df_rf_coef, new_data], ignore_index=True)
    df_rf_coef_mean_abs = abs(df_rf_coef.mean().to_frame().T)
    apo_impor_resids = []
    mmv_impor_resids = []
    for col in df_rf_coef_mean_abs.columns.values:
        importance = df_rf_coef_mean_abs.at[0, col]
        r1, r2 = extract_residues(col)
        if r1 is None:
            continue
        apo_impor_resids.append([r1, importance])
        apo_impor_resids.append([r2, importance])

        mmv_impor_resids.append([r1, importance])
        mmv_impor_resids.append([r2, importance])
    rf_apo_impor_resids_sum = sum_elements(apo_impor_resids)
    rf_mmv_impor_resids_sum = sum_elements(mmv_impor_resids)
    RF_impo_res_apo = pd.concat([RF_impo_res_apo, pd.DataFrame(np.reshape(rf_apo_impor_resids_sum[:, 1], (1, len(rf_apo_impor_resids_sum))), columns=rf_apo_impor_resids_sum[:, 0])], ignore_index=True)
    RF_impo_res_mmv = pd.concat([RF_impo_res_mmv, pd.DataFrame(np.reshape(rf_mmv_impor_resids_sum[:, 1], (1, len(rf_mmv_impor_resids_sum))), columns=rf_mmv_impor_resids_sum[:, 0])], ignore_index=True)
RF_impo_res_apo = RF_impo_res_apo.fillna(0)
RF_impo_res_mmv = RF_impo_res_mmv.fillna(0)

#Plot accuracy calculated from training the earlier model
x = np.linspace(0, len(acc), len(acc))
fig = plt.subplots(figsize=(8,2))
plt.scatter(x,acc, color='k', s=20)
plt.ylim(0,1.1)
```
**MLP training and testing**
```python
acc = []
for run in range(0,20):
    X = df_dropped.iloc[:,:-1]
    y = df_dropped['State']
    if type(X) is not np.ndarray:
        X, y = X.to_numpy(),y.to_numpy()
    else:
        print('X,y already converted to ndarray')
    one_hot_y = y[:,None] == np.array([1,2])
    X_train, X_test, y_train, y_test = train_test_split(X,one_hot_y,test_size = 0.20, random_state = 666) 
    clf = MLP.fit(X_train, y_train)
    y_pred = MLP.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))
    W = clf.coefs_
    B = clf.intercepts_
    L = len(W)   
    A = [X]+[None]*L
    for l in range(L):
        A[l+1] = np.maximum(0,A[l].dot(W[l])+B[l])
    R = [None]*L + [A[L]*one_hot_y]
    for l in range(0,L)[::-1]:    
        w = W[l]
        b = B[l]     
        z = A[l].dot(w)+b
        s = R[l+1] / z               
        c = s.dot(w.T)               
        R[l] = A[l]*c       
    df_mlp_coef = pd.DataFrame(np.reshape(R[0].mean(axis=0), (1,X.shape[1])), columns = df_dropped.columns[:-1])
    apo_impor_resids = []
    mmv_impor_resids = []
    for col in df_rf_coef_mean_abs.columns.values:
        importance = df_rf_coef_mean_abs.at[0, col]
        r1, r2 = extract_residues(col)
        if r1 is None:
            continue
        apo_impor_resids.append([r1, importance])
        apo_impor_resids.append([r2, importance])

        mmv_impor_resids.append([r1, importance])
        mmv_impor_resids.append([r2, importance])
    mlp_apo_impor_resids_sum = sum_elements(apo_impor_resids)
    mlp_mmv_impor_resids_sum = sum_elements(mmv_impor_resids)

    mlp_impo_res_apo = pd.concat([mlp_impo_res_apo, pd.DataFrame(np.reshape(mlp_apo_impor_resids_sum[:, 1], (1, len(mlp_apo_impor_resids_sum))), columns=mlp_apo_impor_resids_sum[:, 0])], ignore_index=True)
    mlp_impo_res_mmv = pd.concat([mlp_impo_res_mmv, pd.DataFrame(np.reshape(mlp_mmv_impor_resids_sum[:, 1], (1, len(mlp_mmv_impor_resids_sum))), columns=mlp_mmv_impor_resids_sum[:, 0])], ignore_index=True)
mlp_impo_res_apo = mlp_impo_res_apo.fillna(0)
mlp_impo_res_mmv = mlp_impo_res_mmv.fillna(0)

#Plot accuracy calculated from training the earlier model
x = np.linspace(0, len(acc), len(acc))
fig = plt.subplots(figsize=(8,2))
plt.scatter(x,acc, color='k', s=20)
plt.ylim(0,1.1)
```

**Visualize the residue Importance**
```python
LR_impo_res_apo.mean().plot(figsize = (10,3), linewidth=2.0, color= 'red', label='LR')
RF_impo_res_mmv.mean().plot(figsize = (10,3), linewidth=2.0, color='orange', label='RF')
mlp_impo_res_apo.mean().plot(figsize = (10,3), linewidth=2.0, color = 'black', label='MLP') #xlim=(390, 519),
plt.xlabel('Residue number', fontsize=12, fontweight='bold')
plt.ylabel('Residue importance', fontsize=12, fontweight='bold')
plt.legend(frameon=True, edgecolor='k', framealpha=1)
#plt.savefig('Residue_importance.png', dpi=600, bbox_inches='tight')

```

**Print out the top residues from each model training**
```python
imp = LR_impo_res_apo.mean()
omp = RF_impo_res_apo.mean()
amp = mlp_impo_res_apo.mean()
top_lr  = imp.sort_values(ascending=False).head(15)
top_rf  = omp.sort_values(ascending=False).head(15)
top_mlp  = amp.sort_values(ascending=False).head(15)
all_res = sorted(set(top_lr.index).union(top_rf.index).union(top_mlp.index))
table = pd.DataFrame(index=all_res, columns=["LR_importance", "RF_importance"])
table.loc[top_lr.index, "LR_importance"] = top_lr.values
table.loc[top_rf.index, "RF_importance"] = top_rf.values
table.loc[top_mlp.index, "MLP_importance"] = top_mlp.values
print(table)
print(all_res)
print("Top LR residues:\n", top_lr)
print("\nTop RF residues:\n", top_rf)
print("Top mlp residues:\n", top_mlp)
```
