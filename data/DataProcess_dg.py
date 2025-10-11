import argparse
import os
import json
import pandas as pd
from openpyxl import load_workbook
from functions.Helper import data_interpolation, data_visualization, alphabet_to_number

# ----------------- args -----------------
parser = argparse.ArgumentParser(description="Process a DG workbook.")
parser.add_argument("FILE", help='Excel filename (e.g., "dgXXX.xlsx"). '
                                 'If only a filename is given, it will be looked up under data/raw_data/.')
args = parser.parse_args()
FILE = args.FILE
# Normalize name and path
basename = os.path.basename(FILE)
name, ext = os.path.splitext(basename)
if not name.lower().startswith("dg"):
    raise ValueError(f'Filename must start with "dg": got "{name}"')

# If user supplied only a filename, prefix the standard folder
if os.path.isabs(FILE) or os.path.sep in FILE:
    file_path = FILE
else:
    file_path = os.path.join("data", "raw_data", FILE)

if not os.path.exists(file_path):
    raise FileNotFoundError(f'Input file not found: {file_path}')

# ----------------- config -----------------
with open("data/functions/config.json", "r") as f:
    config = json.load(f)

# Allow keys to be either the full path's basename or the exact FILE string
cfg_key = basename if basename in config else FILE
if cfg_key not in config:
    raise KeyError(f'Config for "{basename}" not found in data/functions/config.json')

column_names = config[basename]["column_names"]
use_cols = config[basename]["use_cols"]
drop_cols = config[basename]["drop_cols"]
OUTPUT = config[basename]["output"]

# ----------------- Read Excel -----------------
xls = pd.ExcelFile(file_path)
dfs = []
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=4, header=None, usecols = use_cols)
    print(sheet_name)
    print(df.head())
    df.columns = column_names
    df = df.drop(columns=drop_cols, errors="ignore")
    dfs.append(df)

# ----------------- Concatenate & Filter -----------------
concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
filtered_df = concatenated_df[~concatenated_df['Date'].astype(str).str.contains('평균|최대|최소', na=False)]
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%Y-%m-%d', errors='coerce')

# ----------------- write -----------------
out_path = os.path.join("data", "processed_data", OUTPUT)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
filtered_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")


# interpolated_df = data_interpolation(filtered_df)

# interpolated_df['BPR'] = interpolated_df['BP'] / (interpolated_df['M1_in'] * interpolated_df['PS_VS']) * 1000
# interpolated_df['MY'] = interpolated_df['BPR'] * interpolated_df['CH4'] / 100

# interpolated_df.head()
# interpolated_df.drop(columns=['BP', 'BPR', 'CH4'], inplace=True)
# interpolated_df['Date'] = pd.to_datetime(interpolated_df['Date'], format='%Y-%m-%d', errors='coerce')
# interpolated_df.index = interpolated_df['Date']
# interpolated_df.drop(columns=['Date'], inplace=True)
# interpolated_df.to_csv(OUTPUT)

# df = pd.read_csv(OUTPUT, parse_dates=['Date'], index_col=0)
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
# data_visualization(df)

