# Model Training and Evalaution

Load Modules
```python
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import (recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve,classification_report)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
```

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
# Create dataframe to store test set performance
results = {'f1': [], 'precision': [], 'recall': [], 'acc': [], 'roc_auc': [], 'run': [], 'bootstrap': [], 'model': []}

# Store feature importance for each bootstrap
LR_impo_res_apo = pd.DataFrame()
LR_impo_res_mmv = pd.DataFrame()

for run in range(20):
    # Prepare data
    X = df_dropped.iloc[:, :-1].to_numpy()
    y = df_dropped['State'].to_numpy()
    num_features = X.shape[1]
    
    # Single train/test split per run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=run, stratify=y
    )
    
    # Bootstrap training 10 times 
    for bootstrap in range(10):
        X_train_resampl, y_train_resampl = resample(
            X_train, y_train,
            n_samples=len(X_train),
            random_state=bootstrap + run*10  # Modified to ensure unique seeds
        )
        
        y_train_resampl = y_train_resampl.astype(int)
        
        # Train LR
        LR.fit(X_train_resampl, y_train_resampl.ravel())
        
        # Predict
        y_pred = LR.predict(X_test)
        y_pred = y_pred.astype(int)
        
        # Store metrics for this bootstrap
        results['f1'].append(f1_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred))
        results['recall'].append(recall_score(y_test, y_pred))
        results['acc'].append(accuracy_score(y_test, y_pred))
        results['roc_auc'].append(roc_auc_score(y_test, y_pred))
        results['run'].append(run)
        results['bootstrap'].append(bootstrap)
        results['model'].append('LR')

        # Get coefficients as feature importance
        coef_importance = np.abs(LR.coef_).mean(axis=0) if len(LR.coef_.shape) > 1 else np.abs(LR.coef_)
        
        bootstrap_importances = pd.DataFrame(
            np.reshape(coef_importance, (1, num_features)),
            columns=df_dropped.columns[:-1]
        )
        
        # Map to residues
        apo_importances = []
        mmv_importances = []
        
        for col in bootstrap_importances.columns:
            importance = bootstrap_importances[col].iloc[0]
            r1, r2 = extract_residues(col)
            
            if r1 is not None:
                apo_importances.append([r1, importance])
                apo_importances.append([r2, importance])
                mmv_importances.append([r1, importance])
                mmv_importances.append([r2, importance])
        
        # Sum per residue
        lr_apo_sum = sum_elements(apo_importances)
        lr_mmv_sum = sum_elements(mmv_importances)
        
        # Append to DataFrames
        LR_impo_res_apo = pd.concat([
            LR_impo_res_apo,
            pd.DataFrame(
                np.reshape(lr_apo_sum[:, 1], (1, len(lr_apo_sum))),
                columns=lr_apo_sum[:, 0]
            )
        ], ignore_index=True)
        
        LR_impo_res_mmv = pd.concat([
            LR_impo_res_mmv,
            pd.DataFrame(
                np.reshape(lr_mmv_sum[:, 1], (1, len(lr_mmv_sum))),
                columns=lr_mmv_sum[:, 0]
            )
        ], ignore_index=True)

lr_results_df = pd.DataFrame(results)
LR_impo_res_apo = LR_impo_res_apo.fillna(0)
LR_impo_res_mmv = LR_impo_res_mmv.fillna(0)
LR_impo_res_apo.head()
```

**RF Training and Testing**
```python
# Create dataframe to store test set performance
results = {'f1': [], 'precision': [], 'recall': [], 'acc': [], 'roc_auc': [], 'run': [], 'bootstrap': [], 'model': []}

# Store feature importance for each bootstrap
RF_impo_res_apo = pd.DataFrame()
RF_impo_res_mmv = pd.DataFrame()

for run in range(20):
    # Prepare data
    X = df_dropped.iloc[:, :-1].to_numpy()
    y = df_dropped['State'].to_numpy()
    num_features = X.shape[1]
    
    # Single train/test split per run 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=run, stratify=y
    )
    
    # Bootstrap training 
    for bootstrap in range(10):  # Using 10 bootstraps like LR
        X_train_resampl, y_train_resampl = resample(
            X_train, y_train,
            n_samples=len(X_train),
            random_state=bootstrap + run*10  # Unique seed per bootstrap
        )
        
        y_train_resampl = y_train_resampl.astype(int)
        
        # Train RF
        RF.fit(X_train_resampl, y_train_resampl.ravel())
        
        # Predict
        y_pred = RF.predict(X_test)
        y_pred = y_pred.astype(int)
        
        # Store metrics for this bootstrap
        results['f1'].append(f1_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred))
        results['recall'].append(recall_score(y_test, y_pred))
        results['acc'].append(accuracy_score(y_test, y_pred))
        results['roc_auc'].append(roc_auc_score(y_test, y_pred))
        results['run'].append(run)
        results['bootstrap'].append(bootstrap)
        results['model'].append('RF')
        
        # Store feature importances for this bootstrap
        bootstrap_importances = pd.DataFrame(
            np.reshape(RF.feature_importances_, (1, num_features)),
            columns=df_dropped.columns[:-1]
        )
        
        # Convert to residue-level importance
        apo_importances = []
        mmv_importances = []
        
        for col in bootstrap_importances.columns:
            importance = bootstrap_importances[col].iloc[0]
            r1, r2 = extract_residues(col)
            
            if r1 is not None:
                apo_importances.append([r1, importance])
                apo_importances.append([r2, importance])
                mmv_importances.append([r1, importance])
                mmv_importances.append([r2, importance])
        
        # Sum importances per residue
        rf_apo_sum = sum_elements(apo_importances)
        rf_mmv_sum = sum_elements(mmv_importances)
        
        # Store residue importances for this bootstrap
        RF_impo_res_apo = pd.concat([
            RF_impo_res_apo,
            pd.DataFrame(
                np.reshape(rf_apo_sum[:, 1], (1, len(rf_apo_sum))),
                columns=rf_apo_sum[:, 0]
            )
        ], ignore_index=True)
        
        RF_impo_res_mmv = pd.concat([
            RF_impo_res_mmv,
            pd.DataFrame(
                np.reshape(rf_mmv_sum[:, 1], (1, len(rf_mmv_sum))),
                columns=rf_mmv_sum[:, 0]
            )
        ], ignore_index=True)

rf_results_df = pd.DataFrame(results)
RF_impo_res_apo = RF_impo_res_apo.fillna(0)
RF_impo_res_mmv = RF_impo_res_mmv.fillna(0)
RF_impo_res_apo.head()
```
**MLP training and testing**
```python
# Create dataframe to store test set performance
results = {'acc': [], 'run': [], 'bootstrap': [], 'model': []}

# Store feature importance for each bootstrap
mlp_impo_res_apo = pd.DataFrame()
mlp_impo_res_mmv = pd.DataFrame()

for run in range(20):
    # Prepare data
    X = df_dropped.iloc[:, :-1].to_numpy()
    y = df_dropped['State'].to_numpy()
    
    # Convert to one-hot
    one_hot_y = (y[:, None] == np.array([1, 2])).astype(int)
    
    # Single train/test split per run
    X_train, X_test, y_train, y_test = train_test_split(
        X, one_hot_y, test_size=0.20, random_state=run, stratify=y
    )
    
    # Bootstrap training
    for bootstrap in range(10):  # Adding 10 bootstraps for consistency
        X_train_resampl, y_train_resampl = resample(
            X_train, y_train,
            n_samples=len(X_train),
            random_state=bootstrap + run*10
        )
        
        # Train MLP
        clf = MLP.fit(X_train_resampl, y_train_resampl)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        results['acc'].append(accuracy_score(y_test_labels, y_pred_labels))
        results['run'].append(run)
        results['bootstrap'].append(bootstrap)
        results['model'].append('MLP')
        
        # Layer-Wise Relevance Propagation
        W = clf.coefs_
        B = clf.intercepts_
        L = len(W)
        
        # Forward pass
        A = [X_test] + [None] * L
        for l in range(L):
            A[l+1] = np.maximum(0, A[l].dot(W[l]) + B[l])
        
        # Initialize relevance at output
        R = [None] * L + [A[L] * y_test]
        
        # LRP backward pass
        for l in range(L-1, -1, -1):
            w = W[l]
            b = B[l]
            z = A[l].dot(w) + b
            z = z + np.sign(z) * 1e-9
            s = R[l+1] / z
            c = s.dot(w.T)
            R[l] = A[l] * c
        
        # Get feature importance
        feature_importance = np.abs(R[0]).mean(axis=0)
        
        # Create DataFrame for this bootstrap
        bootstrap_importances = pd.DataFrame(
            np.reshape(feature_importance, (1, X.shape[1])),
            columns=df_dropped.columns[:-1]
        )
        
        # Convert to residue-level importance
        apo_importances = []
        mmv_importances = []
        
        for col in bootstrap_importances.columns:
            importance = bootstrap_importances[col].iloc[0]
            r1, r2 = extract_residues(col)
            
            if r1 is not None:
                apo_importances.append([r1, importance])
                apo_importances.append([r2, importance])
                mmv_importances.append([r1, importance])
                mmv_importances.append([r2, importance])
        
        # Sum importances per residue
        mlp_apo_sum = sum_elements(apo_importances)
        mlp_mmv_sum = sum_elements(mmv_importances)
        
        # Store residue importances for this bootstrap
        mlp_impo_res_apo = pd.concat([
            mlp_impo_res_apo,
            pd.DataFrame(
                np.reshape(mlp_apo_sum[:, 1], (1, len(mlp_apo_sum))),
                columns=mlp_apo_sum[:, 0]
            )
        ], ignore_index=True)
        
        mlp_impo_res_mmv = pd.concat([
            mlp_impo_res_mmv,
            pd.DataFrame(
                np.reshape(mlp_mmv_sum[:, 1], (1, len(mlp_mmv_sum))),
                columns=mlp_mmv_sum[:, 0]
            )
        ], ignore_index=True)

mlp_results_df = pd.DataFrame(results)
mlp_impo_res_apo = mlp_impo_res_apo.fillna(0)
mlp_impo_res_mmv = mlp_impo_res_mmv.fillna(0)
mlp_impo_res_mmv.head()
```
**The performance of models** 

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = ['RF', 'MLP', 'LR']
colors = ['blue', 'red', 'green']

for idx, (model, color) in enumerate(zip(models, colors)):
    model_data = all_results[all_results['model'] == model]
    
    # Create x values (just the index/order of runs)
    x = np.arange(len(model_data))
    
    # Scatter plot
    axes[idx].scatter(x, model_data['acc'], color=color, s=20, alpha=0.6)
    axes[idx].set_ylim(0, 1.1)
    axes[idx].set_title(f'{model} Accuracy Distribution')
    axes[idx].set_xlabel('Sample (Run × Bootstrap)')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].grid(True, alpha=0.3)

    mean_acc = model_data['acc'].mean()
    axes[idx].axhline(y=mean_acc, color='black', linestyle='--', alpha=0.5, label=f'Mean: {mean_acc:.3f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('Model_Performance.png', dpi=200, bbox_inches='tight')
plt.show()
```
