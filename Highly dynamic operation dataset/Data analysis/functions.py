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

def current_health_state(diag_folder: str, cap_file: str, reshape_dims=(8, 25), final_capacity_divisor=3.2):
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
        if hidden.size(1) != self.num_gru_layers:
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
    
def load_trained_model(model_path, input_dim_encoder, hidden_dim, output_dim, n_cat, num_encoder_fc_layers, num_gru_layers, num_decoder_fc_layers, device):
    encoder = Encoder(input_dim_encoder, hidden_dim, num_fc_layers=num_encoder_fc_layers).to(device)
    decoder = Decoder(hidden_dim, output_dim, n_cat, num_gru_layers=num_gru_layers, num_fc_layers=num_decoder_fc_layers).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_new_data(model, loader, criterion, device,n):
    model.eval()
    all_outputs = []
    all_targets = []
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

def prediction(diagcap_final, sequence, futureVR, input_dim_encoder, hidden_dim, output_dim, n_cat, num_encoder_fc_layers, num_gru_layers, num_decoder_fc_layers, device, batch_size, mode, result_folder, model_path='Best model.pt'):
    final_results = []  
    all_fold_results = [] 

    for n, ref in zip(range(1, 11), range(1, 11)):
        concat_diag = []
        concat_seq = []
        concat_VR = []

        for t in range(0, 25 - ref):
            sliced_diag = diagcap_final[:, t, :]
            concat_diag.append(sliced_diag)

            sliced_seq = sequence[:, t:t + n]
            concat_seq.append(sliced_seq)

            sliced_VR = futureVR[:, t + 1:t + n + 1]
            concat_VR.append(sliced_VR)

        diag_all = torch.cat(concat_diag, dim=0)
        seq_all = torch.cat(concat_seq, dim=0)
        VR_all = torch.cat(concat_VR, dim=0)

        val_dataset = TensorDataset(diag_all, seq_all, VR_all)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        encoder = Encoder(input_dim_encoder, hidden_dim, num_fc_layers=num_encoder_fc_layers).to(device)
        decoder = Decoder(hidden_dim, output_dim, n_cat, num_gru_layers=num_gru_layers, num_fc_layers=num_decoder_fc_layers).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        criterion = nn.MSELoss()

        fold_rmse = []
        fold_mape = []
        
        model_path = 'Best model.pt'   

        model = load_trained_model(model_path, input_dim_encoder, hidden_dim, output_dim, n_cat, num_encoder_fc_layers, num_gru_layers, num_decoder_fc_layers, device)
        rmse_by_time, mape_by_time, all_outputs, all_targets= evaluate_new_data(model, val_loader, criterion, device, n)

        rmse_last = rmse_by_time[-1]
        mape_last = mape_by_time[-1]

        fold_rmse.append(rmse_last)
        fold_mape.append(mape_last)

        # Store individual fold results for tracking purposes
        all_fold_results.append({
            'n': n,
            'ref': ref,
            'RMSE': rmse_last,
            'MAPE': mape_last,
            'model' : model_path
        })
        
        predictions_list = [all_outputs.numpy()]
        targets_list = [all_targets.numpy()]
        
        with pd.ExcelWriter(f'{result_folder}/seq2seq_predictions_sheet_n{n}_ref{ref}.xlsx') as writer:
            for index, predictions in enumerate(predictions_list):
                for sheet_index in range(predictions.shape[0]):
                    df_predictions = pd.DataFrame(predictions[sheet_index, :, :])
                    sheet_name = f'Sheet{sheet_index+1}'
                    df_predictions.to_excel(writer, sheet_name=sheet_name, index=False)

        with pd.ExcelWriter(f'{result_folder}/seq2seq_targets_sheet_n{n}_ref{ref}.xlsx') as writer:
            for index, predictions in enumerate(targets_list):
                for sheet_index in range(predictions.shape[0]):
                    df_predictions = pd.DataFrame(predictions[sheet_index, :, :])
                    sheet_name = f'Sheet{sheet_index+1}'
                    df_predictions.to_excel(writer, sheet_name=sheet_name, index=False)            
                    
        # Calculate the average RMSE and MAPE across all folds
        avg_rmse = np.mean(fold_rmse)
        avg_mape = np.mean(fold_mape)
        
        print(f'n = {n}, avg_mape = {avg_mape}')

        # Append the average results for this (n, ref) combination to the list
        final_results.append({
            'n': n,
            'ref': ref,
            'Average_RMSE': avg_rmse,
            'Average_MAPE': avg_mape,
            'model' : model_path
        })

    df_final_results = pd.DataFrame(final_results)
    csv_file_avg = f'{result_folder}/fold_predictions_avg_{mode}.csv'
    df_final_results.to_csv(csv_file_avg, index=False)

    df_all_fold_results = pd.DataFrame(all_fold_results)
    csv_file_all_folds = f'{result_folder}/fold_predictions_all_{mode}.csv'
    df_all_fold_results.to_csv(csv_file_all_folds, index=False)


def target_capacity(capacity, num_folds, randomst, capacity_folder):
    for n, ref in zip(range(1, 11), range(1, 11)):
        set_seed(0)
        concat_capacity_all = []
        for t in range(0,25-ref) :
            set_seed(0)
            sliced_capacity_all = capacity[:,t:t+n]
            concat_capacity_all.append(sliced_capacity_all)

        capacity_all = torch.cat(concat_capacity_all, dim = 0)
        best_predictions = capacity_all.numpy()
        df_best_predictions = pd.DataFrame(best_predictions.reshape(best_predictions.shape[0], -1))
        df_best_predictions.to_csv(f'{capacity_folder}/capacity_n{n}_ref{ref}.csv', index=False)

def unified_regression(result_folder, capacity_folder, n_folds, n_true, ref_true):

    true_all = []
    capacity_all = []
    true_file = f'{result_folder}/seq2seq_targets_sheet_n{n_true}_ref{ref_true}.xlsx'
    sheets_dict = pd.read_excel(true_file, sheet_name=None, header=None)
    sheet_names = [f'Sheet{j}' for j in range(1, 8*(25-n_true)+1)]
    data_frames = []
    for sheet_name in sheet_names:
        if sheet_name in sheets_dict:
            selected_row = sheets_dict[sheet_name].iloc[1:n_true+1, :10]
            data_frames.append(selected_row)
    true_data = pd.concat(data_frames).reset_index(drop=True)   
    true_all.append(true_data)
    
    cap_file = f'{capacity_folder}/capacity_n{n_true}_ref{ref_true}.csv'
    cap_data = pd.read_csv(cap_file, header=None).iloc[1:8*(25-n_true)+1, :n_true]
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
        n_estimators=100,      
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

def evaluate_predictions(result_folder, capacity_folder, model):
    final_rmse_scores = []
    final_mape_scores = []

    for n, ref in zip(range(1, 11), range(1, 11)):
        predictions_all = []
        capacity_all = []
        
        pred_file = f'{result_folder}/seq2seq_predictions_sheet_n{n}_ref{ref}.xlsx'
        sheets_dict = pd.read_excel(pred_file, sheet_name=None, header=None)
        sheet_names = [f'Sheet{j}' for j in range(1, 8*(25-n)+1)]
        data_frames = []
        
        for sheet_name in sheet_names:
            if sheet_name in sheets_dict:
                selected_row = sheets_dict[sheet_name].iloc[1:n+1, :10]
                data_frames.append(selected_row)
        pred_data = pd.concat(data_frames).reset_index(drop=True)  
        predictions_all.append(pred_data)
        
        cap_file = f'{capacity_folder}/capacity_n{n}_ref{ref}.csv'
        cap_data = pd.read_csv(cap_file, header=None).iloc[1:8*(25-n)+1, :n]
        cap_data = cap_data.stack().reset_index(drop=True)
        capacity_all.append(cap_data)

        predictions_all = pd.concat(predictions_all, ignore_index=True)
        capacity_all = pd.concat(capacity_all, ignore_index=True)

        n_folds_pred = 4
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

            rmse_scores_pred.append(rmse)
            mape_scores_pred.append(mape)
            fold_results_pred.append((y_test_pred, y_pred_pred))

        avg_rmse = np.mean(rmse_scores_pred)
        avg_mape = np.mean(mape_scores_pred)
        
        print(f'average mape for n = {n} : {avg_mape}')
        
        final_rmse_scores.append(avg_rmse)
        final_mape_scores.append(avg_mape)

        performance_df = pd.DataFrame({'Fold': range(1, n_folds_pred + 1), 'RMSE': rmse_scores_pred, 'MAPE': mape_scores_pred})
        performance_df.to_csv(f'{result_folder}/fold_performance_pred_rf_n{n}_ref{ref}.csv', index=False)

        with pd.ExcelWriter(f'{result_folder}/predicted_results_pred_rf_n{n}_ref{ref}.xlsx') as writer:
            for i, (y_true, y_pred) in enumerate(fold_results_pred, start=1):
                result_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
                result_df.to_excel(writer, sheet_name=f'Fold {i}')
    
    final_performance_df = pd.DataFrame({'n': range(1, 11), 'Average_RMSE': final_rmse_scores, 'Average_MAPE': final_mape_scores})
    final_performance_df.to_csv(f'{result_folder}/overall_average_performance.csv', index=False)