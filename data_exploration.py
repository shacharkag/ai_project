import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



def print_missing_data():
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        print(data.columns.values)
        msno.matrix(data)
        data.apply(lambda x: 43 - x.count(), axis=1).value_counts()


def find_correlation():
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        train, test = train_test_split(data, train_size=0.8, random_state=11)
        corr_matrix = train.corr()
        fig, ax = plt.subplots(figsize=(18, 13))
        sns.heatmap(corr_matrix, annot_kws={"size": 20})
        plt.title(year)
        plt.show()

        kot = corr_matrix[corr_matrix >= .9]
        plt.figure(figsize=(12, 10))
        sns.heatmap(kot, cmap="Greens")
        plt.title(f'{year} : corr_matrix >= .9')
        plt.show()

        g = sns.jointplot(data=train, x="p1_svpt", y="p2_2ndWon", hue="p1_won")
        g.ax_joint.grid()
        plt.show()


        g = sns.jointplot(data=train, x="p1_elo_bestRank", y="p1_elo_rank", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        g = sns.jointplot(data=train, x="p2_elo_bestRank", y="p2_elo_rank", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        g = sns.jointplot(data=train, x="p2_elo_bestRank", y="p2_elo_rank", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        g = sns.jointplot(data=train, x="p1_elo_points", y="p1_elo_bestPoints", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        g = sns.jointplot(data=train, x="p2_elo_points", y="p2_elo_rank", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        g = sns.jointplot(data=train, x="p1_elo_points", y="p1_elo_rank", hue="p1_won")
        g.ax_joint.grid()
        plt.show()

        print(f'{year}:')
        get_top_abs_correlations(train)


def get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack().drop_duplicates()
    print(au_corr[au_corr > 0.8])
    return au_corr


def print_all_features():
    set_of_cols = set()
    for year in range(2001, 2022):
        tbl = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        set_of_cols.update(set(tbl.columns.values))
    print(set_of_cols)


def main():
    print_all_features()
    find_correlation()


if __name__ == '__main__':
    main()