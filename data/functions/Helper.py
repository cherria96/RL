from scipy.interpolate import CubicSpline, PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt


def data_interpolation(filtered_df):

    # Make a copy of dataframe to interpolate
    interpolated_df = filtered_df.copy()

    # For each numeric column, interpolate missing values using monotonic cubic spline (PCHIP is monotonic alternative to CubicSpline)
    # Identify numeric columns (except Date)
    numeric_cols = interpolated_df.select_dtypes(include='number').columns

    # Interpolate each column
    for col in numeric_cols:
        # Get valid indices
        valid = ~interpolated_df[col].isna()
        if valid.sum() >= 2:  # Need at least 2 points to interpolate
            x = interpolated_df.loc[valid, 'Date'].map(pd.Timestamp.toordinal)
            y = interpolated_df.loc[valid, col]
            interpolator = PchipInterpolator(x, y)
            
            # Interpolate missing values
            missing = interpolated_df[col].isna()
            interpolated_df.loc[missing, col] = interpolator(interpolated_df.loc[missing, 'Date'].map(pd.Timestamp.toordinal))

def data_visualization(df):
    # Plot MY trend over time
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['MY'], marker='o', linestyle='-', alpha=0.7)
    plt.title("Trend of MY Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("MY", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def alphabet_to_number(s: str) -> int:
    """
    Convert Excel-style column letters (A, B, ..., Z, AA, AB, ...) to numbers.
    """
    result = 0
    for char in s:
        result = result * 26 + (ord(char.upper()) - ord('A')+1)
    return result - 1

# Examples:
if __name__ == "__main__":
    use_cols ="B:BH"
    start,end = use_cols.split(":")
    print(alphabet_to_number(end))   # 1
    print(alphabet_to_number(start))   # 26
