from train import load_data_from_hbase, postprocess_result_data_from_hbase, save_data_from_list_keys
import sys
import pandas as pd
import os

class Predict():
    """
        Сlass that allows you make prediction cluster for index from data_result_from_hbase_with_postprocess.csv
    """

    @staticmethod
    def load_result_data(name_table_result='/home/yagor/Рабочий стол/mipt/lab3/notebook/data_result_from_hbase_with_postprocess.csv'):
        return pd.read_csv(name_table_result, index_col=['index'])
    
    @staticmethod
    def predict(data_from_db_new_res, index: int=8):
        """
                Class method which make predict for index
            Args:
                data_from_db_new_res pd.Dataframe: [index, prediction] columns
                index int: index for which make prediction
            Returns:
                cluster: cluster for the index
        """
        try:
            cluster = data_from_db_new_res.loc[index, 'prediction']
            return cluster
        except:
            print('For this index cluster not defind')


if __name__ == "__main__":
    try:
        name_table_result = str(sys.argv[1])
        index = int(sys.srgv[2])
    except:
        print('Введите корректные входные данные')
        sys.exit(1)
    data_from_db_new_res = Predict.load_result_data(name_table_result)
    cluster = Predict.predict(data_from_db_new_res, index)
    print(f"Кластер у индекса {index}: {cluster}")