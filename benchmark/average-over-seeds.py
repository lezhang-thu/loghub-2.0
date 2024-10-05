import sys
import os
import pandas as pd

if __name__ == '__main__':
    runs = int(sys.argv[1])
    common_file = sys.argv[2]
    ret_file = sys.argv[3]

    prefix = os.path.dirname(common_file)[:-1]
    common_file = os.path.basename(common_file)

    all_files = []
    header_row = None
    for k in range(runs):
        file_name = '{}{}'.format(prefix, k)
        file_name = os.path.join(file_name, common_file)
        df = pd.read_csv(file_name)
        df.set_index(df.columns[0], inplace=True)
        if header_row is None:
            header_row = df.iloc[0]
        df = df.loc[['GA', 'PA', 'FGA', 'FTA']]
        all_files.append(df)

    all_files = [df.apply(pd.to_numeric) for df in all_files]
    combined_df = pd.concat(all_files, axis=0)
    average_df = combined_df.groupby(level=0).mean().round(3)
    average_df = pd.concat([header_row.to_frame().T, average_df],
                           ignore_index=False)
    average_df = average_df.loc[['Dataset', 'GA', 'PA', 'FGA', 'FTA']]
    average_df.T.to_csv(ret_file)
