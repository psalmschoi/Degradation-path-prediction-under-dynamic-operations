import os
import random
import sys
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array, check_random_state
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits

# GroupKFold
class GroupKFold(_BaseKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        groups = check_array(groups, ensure_2d=False, dtype=None)
        unique_groups, groups = np.unique(groups, return_inverse=True)
        num_groups = len(unique_groups)
        samples_per_group = np.bincount(groups)
        sorted_indices = np.argsort(samples_per_group)[::-1]

        if self.shuffle:
            rng = check_random_state(self.random_state)
            for sample_count in np.unique(samples_per_group):
                same_count_indices = np.where(samples_per_group == sample_count)[0]
                chunk = sorted_indices[same_count_indices]
                rng.shuffle(chunk)
                sorted_indices[same_count_indices] = chunk

        samples_per_group = samples_per_group[sorted_indices]
        samples_per_fold = np.zeros(self.n_splits)
        group_to_fold = np.zeros(len(unique_groups))

        for group_idx, weight in enumerate(samples_per_group):
            lightest = np.argmin(samples_per_fold)
            samples_per_fold[lightest] += weight
            group_to_fold[sorted_indices[group_idx]] = lightest

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)                

# Current state tensor
def read_csv_file(file_path):
    data = pd.read_csv(file_path, header=None).values
    return data

def process_folder(folder_path):
    tensor_list = []
    file_names = sorted(os.listdir(folder_path))

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        data = read_csv_file(file_path)
        unsqueezed_data = np.expand_dims(data, axis=-1)
        tensor_list.append(unsqueezed_data)
        
    final_tensor = np.stack(tensor_list, axis=0)
    return torch.tensor(final_tensor, dtype=torch.float32)

def current_health_state(diag_folder: str, cap_file: str, reshape_dims=(72, 25), final_capacity_divisor=3.2):
    diagnosis_tensor = process_folder(diag_folder)
    df = pd.read_excel(cap_file)
    data = df.iloc[:, 2:].values 
    
    tensor_data = np.array(data).reshape(*reshape_dims)
    output_tensor = np.expand_dims(tensor_data, axis=-1)
    cap_tensor = np.expand_dims(output_tensor, axis=-1)
    
    diagcap_tensor = np.concatenate([diagnosis_tensor, cap_tensor], axis=2)
    diagcap = torch.squeeze(torch.tensor(diagcap_tensor, dtype=torch.float32), dim=-1)
    
    part_diagcap = diagcap[:, :, :10]
    diagcap[:, :, 10] /= final_capacity_divisor

    diagcap_final = torch.cat([diagcap[:, :, 10:], part_diagcap], dim=2)
    return diagcap_final


# future DI tensor
def process_folder_futureDI(folder_path):
    tensor_list = []
    file_names = sorted(os.listdir(folder_path))
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        data = read_csv_file(file_path)
        tensor_list.append(data)
    final_tensor = np.stack(tensor_list, axis=0)
    return torch.tensor(final_tensor, dtype=torch.float32)


# Future health state prediction model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_fc_layers=1):
        super(Encoder, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_fc_layers)])
    def forward(self, x):
        for fc in self.fc_layers:
            x = torch.relu(fc(x))
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_categories, num_gru_layers=1, num_fc_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(n_categories, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_gru_layers, batch_first=True)
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_fc_layers)])
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.num_gru_layers = num_gru_layers

    def forward(self, x, hidden):
        x = self.embedding(x)
        if hidden.size(0) != self.num_gru_layers:
            hidden = hidden.unsqueeze(0).expand(self.num_gru_layers, -1, -1).contiguous()
        output, hidden = self.gru(x, hidden)
        for fc in self.fc_layers:
            output = torch.relu(fc(output))
        prediction = self.fc_output(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        hidden = self.encoder(src)
        outputs, _ = self.decoder(trg, hidden)
        return outputs
    

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        state, conditions, capacity = [item.to(device) for item in batch]
        optimizer.zero_grad()
        output = model(state, conditions)
        diff = output - capacity
        loss = (diff ** 2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, n):
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch in loader:
            state, conditions, VR = [item.to(device) for item in batch]
            outputs = model(state, conditions)
            all_outputs.append(outputs.cpu())
            all_targets.append(VR.cpu())
            
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    rmse_by_time = []
    mape_by_time = []
    
    for future_time in range(1, n + 1):
        mse = ((all_outputs[:, :future_time, :] - all_targets[:, :future_time, :]) ** 2).mean()
        rmse = torch.sqrt(mse)
        mape = torch.mean(torch.abs((all_targets[:, :future_time, :] - all_outputs[:, :future_time, :]) / all_targets[:, :future_time, :])) * 100
        rmse_by_time.append(rmse.item())
        mape_by_time.append(mape.item())
    return rmse_by_time, mape_by_time, all_outputs, all_targets

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate(diagcap_final, sequence, futureVR, groups, num_folds, randomst, 
                       input_dim_encoder, hidden_dim, num_encoder_fc_layers, 
                       n_cat, num_gru_layers, num_decoder_fc_layers, 
                       output_dim, batch_size, lr, num_epochs, early_stopping_patience, 
                       result_folder, device, refs):
    for ref in range(1,refs) :
        for n in range(1,ref+1) :
            kf = GroupKFold(n_splits=num_folds, shuffle = True, random_state = randomst)
            for fold, (train_idx, val_idx) in enumerate(kf.split(diagcap_final, groups=groups)):
                set_seed(0)
                result_prefix = f'fold_{fold+1}'
                diag_train, diag_val = diagcap_final[train_idx], diagcap_final[val_idx]
                seq_train, seq_val = sequence[train_idx], sequence[val_idx]
                VR_train, VR_val = futureVR[train_idx], futureVR[val_idx]
                concat_diag_train = []
                concat_seq_train = []
                concat_VR_train = []
                concat_diag_val = []
                concat_seq_val = []
                concat_VR_val = []
                all_results = []
                results = []
                predictions_list = []
                targets_list = []
                best_mape = float('inf')
                best_epoch = -1
                epochs_no_improve = 0

                for t in range(0,25-ref) :
                    set_seed(0)
                    sliced_diag_train = diag_train[:,t,:]
                    concat_diag_train.append(sliced_diag_train)
                    sliced_diag_val = diag_val[:,t,:]
                    concat_diag_val.append(sliced_diag_val)
                    sliced_seq_train = seq_train[:, t:t+n]
                    concat_seq_train.append(sliced_seq_train)
                    sliced_seq_val = seq_val[:, t:t+n]
                    concat_seq_val.append(sliced_seq_val)
                    sliced_VR_train = VR_train[:, t+1:t+n+1]
                    concat_VR_train.append(sliced_VR_train)
                    sliced_VR_val = VR_val[:, t+1:t+n+1]
                    concat_VR_val.append(sliced_VR_val)

                diag_train_all = torch.cat(concat_diag_train, dim = 0)
                diag_val_all = torch.cat(concat_diag_val, dim = 0)
                seq_train_all = torch.cat(concat_seq_train, dim = 0)
                seq_val_all = torch.cat(concat_seq_val, dim = 0)
                VR_train_all = torch.cat(concat_VR_train, dim = 0)
                VR_val_all = torch.cat(concat_VR_val, dim = 0)
                train_dataset = TensorDataset(diag_train_all, seq_train_all, VR_train_all)
                val_dataset = TensorDataset(diag_val_all, seq_val_all, VR_val_all)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                encoder = Encoder(input_dim_encoder, hidden_dim, num_fc_layers=num_encoder_fc_layers).to(device)
                decoder = Decoder(hidden_dim, output_dim, n_cat, num_gru_layers=num_gru_layers, num_fc_layers=num_decoder_fc_layers).to(device)
                model = Seq2Seq(encoder, decoder, device).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                for epoch in range(num_epochs):
                    train_loss = train(model, train_loader, optimizer, criterion, device)
                    avg_rmse, avg_mape, final_outputs, targets=  evaluate(model, val_loader, criterion, device,n)

                    if epoch % 10 == 0 :
                        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Avg Val RMSE: {avg_rmse[-1]:.4f}, Avg Val mape: {avg_mape[-1]:.4f}")

                    if avg_mape[-1] < best_mape:
                        best_rmse = avg_rmse[-1]
                        best_mape = avg_mape[-1]
                        best_epoch = epoch
                        best_predictions = final_outputs.numpy()
                        best_targets = targets.numpy()
                        results = [(n, avg_rmse, avg_mape)]
                        predictions_list = [best_predictions]
                        targets_list = [best_targets]
                        time_step_rmse = avg_rmse
                        time_step_mape = avg_mape
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), f'{result_folder}/{result_prefix}_best_model_state_dict_n{n}_ref{ref}.pt')                    
                    else:
                        epochs_no_improve += 1
                        
                    if epochs_no_improve >= early_stopping_patience: # Early stopping
                        print(f'Early stopping at epoch {epoch+1}')
                        break
                    
                if best_epoch != -1:
                    result_prefix = f'fold_{fold+1}'
                    all_results.append((best_epoch, n, best_rmse, best_mape))
                    print(f"Completed training for n = {n}, Best Epoch {best_epoch+1}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.4f}.")
                df_results = pd.DataFrame(all_results, columns=['best_epoch', 'n', 'Best RMSE', 'Best MAPE'])
                df_results.to_csv(f'{result_folder}/{result_prefix}_seq2seq_avg_rmse_mape_results_n{n}_ref{ref}.csv', index=False)

                with pd.ExcelWriter(f'{result_folder}/{result_prefix}_seq2seq_predictions_sheet_n{n}_ref{ref}.xlsx') as writer:
                    for index, predictions in enumerate(predictions_list):
                        for sheet_index in range(predictions.shape[0]):
                            df_predictions = pd.DataFrame(predictions[sheet_index, :, :])
                            sheet_name = f'Sheet{sheet_index+1}'
                            df_predictions.to_excel(writer, sheet_name=sheet_name, index=False)
                with pd.ExcelWriter(f'{result_folder}/{result_prefix}_seq2seq_targets_sheet_n{n}_ref{ref}.xlsx') as writer:
                    for index, predictions in enumerate(targets_list):
                        for sheet_index in range(predictions.shape[0]):
                            df_predictions = pd.DataFrame(predictions[sheet_index, :, :])
                            sheet_name = f'Sheet{sheet_index+1}'
                            df_predictions.to_excel(writer, sheet_name=sheet_name, index=False)
    
def target_capacity(capacity, groups, num_folds, randomst, capacity_folder):
    for ref in range(1,11) :
        for n in range(1,ref+1) :
            kf = GroupKFold(n_splits=num_folds, shuffle = True, random_state = randomst)
            for fold, (train_idx, val_idx) in enumerate(kf.split(capacity, groups=groups)):
                set_seed(0)
                capacity_train, capacity_val = capacity[train_idx], capacity[val_idx]
                concat_capacity_train = []
                concat_capacity_val = []
                for t in range(0,25-ref) :
                    set_seed(0)
                    sliced_capacity_train = capacity_train[:,t:t+n]
                    concat_capacity_train.append(sliced_capacity_train)
                    sliced_capacity_val = capacity_val[:,t:t+n]
                    concat_capacity_val.append(sliced_capacity_val)
                capacity_train_all = torch.cat(concat_capacity_train, dim = 0)
                capacity_val_all = torch.cat(concat_capacity_val, dim = 0)            
                result_prefix = f'fold_{fold+1}'
                best_predictions = capacity_val_all.numpy()
                df_best_predictions = pd.DataFrame(best_predictions.reshape(best_predictions.shape[0], -1))
                df_best_predictions.to_csv(f'{capacity_folder}/{result_prefix}_capacity_n{n}_ref{ref}.csv', index=False)

def unified_regression(result_folder, capacity_folder, n_folds, n_true, ref_true):
    true_all = []
    capacity_all = []

    for i in range(1, n_folds+1):
        true_file = f'{result_folder}/fold_{i}_seq2seq_targets_sheet_n{n_true}_ref{ref_true}.xlsx'
        sheets_dict = pd.read_excel(true_file, sheet_name=None, header=None)
        sheet_names = [f'Sheet{j}' for j in range(1, 12*(25-n_true)+1)]
        data_frames = []
        for sheet_name in sheet_names:
            if sheet_name in sheets_dict:
                selected_row = sheets_dict[sheet_name].iloc[1:n_true+1, :10]
                data_frames.append(selected_row)
        true_data = pd.concat(data_frames).reset_index(drop=True)   
        true_all.append(true_data)
        
        cap_file = f'{capacity_folder}/fold_{i}_capacity_n{n_true}_ref{ref_true}.csv'
        cap_data = pd.read_csv(cap_file, header=None).iloc[1:12*(25-n_true)+1, :n_true]
        cap_data = cap_data.stack().reset_index(drop=True)
        capacity_all.append(cap_data)
        
    true_all = pd.concat(true_all, ignore_index=True)
    capacity_all = pd.concat(capacity_all, ignore_index=True)

    kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    rmse_scores = []
    mape_scores = []
    fold_results = []

    for train_index, test_index in kf.split(true_all):
        X_train, X_test = true_all.iloc[train_index], true_all.iloc[test_index]
        y_train, y_test = capacity_all.iloc[train_index], capacity_all.iloc[test_index]
        
        model = RandomForestRegressor(
        n_estimators=400,      
        max_features='sqrt',  
        max_depth=None,          
        random_state=0        
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = ((y_pred - y_test) ** 2).mean()
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        print(mape)

        rmse_scores.append(rmse)
        mape_scores.append(mape)
        fold_results.append((y_test, y_pred))

    performance_df = pd.DataFrame({'Fold': range(1, n_folds+1), 'RMSE': rmse_scores, 'MAPE': mape_scores})
    performance_df.to_csv(f'{result_folder}/fold_performance_true_rf_n{n_true}_ref{ref_true}.csv', index=False)

    with pd.ExcelWriter(f'{result_folder}/predicted_results_true_rf_n{n_true}_ref{ref_true}.xlsx') as writer:
        for i, (y_true, y_pred) in enumerate(fold_results, start=1):
            result_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
            result_df.to_excel(writer, sheet_name=f'Fold {i}')
    return model

def evaluate_predictions(result_folder, capacity_folder, model, n_folds=6):
    for ref in range(1,2) :
        for n in range(1,ref+1) :
            n_folds = 6
            predictions_all = []
            capacity_all = []
            for i in range(1, n_folds+1):
                pred_file = f'{result_folder}/fold_{i}_seq2seq_predictions_sheet_n{n}_ref{ref}.xlsx'
                sheets_dict = pd.read_excel(pred_file, sheet_name=None, header=None)
                sheet_names = [f'Sheet{j}' for j in range(1, 12*(25-n)+1)]
                data_frames = []
                for sheet_name in sheet_names:
                    if sheet_name in sheets_dict:
                        selected_row = sheets_dict[sheet_name].iloc[1:n+1, :10]
                        data_frames.append(selected_row)
                pred_data = pd.concat(data_frames).reset_index(drop=True)  
                predictions_all.append(pred_data)
                
                cap_file = f'{capacity_folder}/fold_{i}_capacity_n{n}_ref{ref}.csv'
                cap_data = pd.read_csv(cap_file, header=None).iloc[1:12*(25-n)+1, :n]
                cap_data = cap_data.stack().reset_index(drop=True)
                capacity_all.append(cap_data)

            predictions_all = pd.concat(predictions_all, ignore_index=True)
            capacity_all = pd.concat(capacity_all, ignore_index=True)

            n_folds_pred = 6
            kf_pred = KFold(n_splits=n_folds_pred, random_state=0, shuffle=True)
            rmse_scores_pred = []
            mape_scores_pred = []
            fold_results_pred = []
            for train_index, test_index in kf_pred.split(predictions_all):
                X_train_pred, X_test_pred = predictions_all.iloc[train_index], predictions_all.iloc[test_index]
                y_train_pred, y_test_pred = capacity_all.iloc[train_index], capacity_all.iloc[test_index]
                y_pred_pred = model.predict(X_test_pred)
                
                mse = ((y_pred_pred - y_test_pred) ** 2).mean()
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test_pred - y_pred_pred) / y_test_pred)) * 100
                print(mape)
            
                rmse_scores_pred.append(rmse)
                mape_scores_pred.append(mape)
                fold_results_pred.append((y_test_pred, y_pred_pred))

            performance_df = pd.DataFrame({'Fold': range(1, n_folds+1), 'RMSE': rmse_scores_pred, 'MAPE': mape_scores_pred})
            performance_df.to_csv(f'{result_folder}/fold_performance_pred_rf_n{n}_ref{ref}.csv', index=False)

            with pd.ExcelWriter(f'{result_folder}/predicted_results_pred_rf_n{n}_ref{ref}.xlsx') as writer:
                for i, (y_true, y_pred) in enumerate(fold_results_pred, start=1):
                    result_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
                    result_df.to_excel(writer, sheet_name=f'Fold {i}')