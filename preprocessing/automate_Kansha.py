from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import fire

def correlation_for_dropping(data, threshold):
    col_corr = set()  # Kumpulkan semua nama kolom yang berkorelasi
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # Dapatkan nama kolom
                col_corr.add(colname)
    return col_corr


def preprocess_data(data_path, target_column, save_path_train, save_path_test):
    df = pd.read_csv(data_path)
    # Menentukan fitur numerik dan kategoris
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    missing_values = df[df.isna().any(axis=1)].index
    columns_to_drop = ['Formatted Date', 'Summary', 'Daily Summary', 'Loud Cover']
    df = df.drop(columns=columns_to_drop)
    df = df.drop(missing_values)
    df = df.drop_duplicates()
    df = pd.get_dummies(df, columns=['Precip Type'])

    X = df.drop('Apparent Temperature (C)', axis=1)
    y = df['Apparent Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    cols_to_drop = correlation_for_dropping(X_train, 0.9)
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Menggabungkan menjadi satu dataset
    X_train_final = pd.DataFrame(X_train_scaled, index=y_train.index)
    train = pd.concat([X_train_final, y_train], axis=1)
    X_test_final = pd.DataFrame(X_test_scaled, index=y_test.index)
    test = pd.concat([X_test_final, y_test], axis=1)

    # Export to csv
    train.to_csv(save_path_train, index=False)
    test.to_csv(save_path_test, index=False)

if __name__ == '__main__':
    fire.Fire(preprocess_data)