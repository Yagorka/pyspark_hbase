import happybase
import csv
import pandas as pd
import sys


def load_data_from_hbase(name_tamle):
    connection = happybase.Connection('hbase-docker', 9090)
    table = connection.table(name_tamle)
    all_data = {}
    for key, data in table.scan():
        all_data[key] = data
    return all_data

def postprocess_train_data_from_hbase(all_data):
    """
            Function which decode and postprocess train data
        Args:
            all_data dict(): dict with dicts rows
        Returns:
            list[dict]: decode data from all_data
    """
    decode_all_data = []
    funcs = {'nutrition_data:exceeded': int, 'nutrition_data:product': str}
    for key in all_data.keys():
        decode_row_data = {}
        for second_key, val in all_data[key].items():
            k = second_key.decode("utf-8")
            v = val.decode("utf-8")
            if k in funcs:
                v = funcs[k](v)
            else:
                v = float(v)
            decode_row_data[k.split(':')[-1]] = v
        decode_all_data.append(decode_row_data)
    return decode_all_data

def postprocess_result_data_from_hbase(all_data):
    """
            Function which decode and postprocess result data
        Args:
            all_data dict(): dict with dicts rows
        Returns:
            list[dict]: decode data from all_data
    """
    decode_all_data = []
    for key in all_data.keys():
        decode_row_data = {}
        for second_key, val in all_data[key].items():
            k = second_key.decode("utf-8")
            v = val.decode("utf-8")
            try:
                cluster = int(v)
                decode_row_data['prediction'] = cluster
            except:
                print(f'key:{k}, value:{v} not int')
        try:
            decode_row_data['index'] = int(key.decode("utf-8"))
        except:
            print(f'key:{key.decode("utf-8")} not int')
        decode_all_data.append(decode_row_data)
    return decode_all_data

def save_data_from_list_keys(decode_all_data, name_csv_for_save : str ='data_from_hbase.csv'):
    """
            Function which saves decode and postprocess result data
        Args:
            decode_all_data dict(): list with dicts rows for train KMeans
            name_csv_for_save str: path for save data
    """
    keys = decode_all_data[0].keys()

    with open(name_csv_for_save, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(decode_all_data)

    
if __name__ == "__main__":
    name_table = sys.argv[1]
    file_for_saves = sys.argv[2]
    train_data = sys.argv[3]
    all_data = load_data_from_hbase(name_table)
    if train_data:
        decode_all_data = postprocess_train_data_from_hbase(all_data)
    else:
        decode_all_data = postprocess_result_data_from_hbase(all_data)
    save_data_from_list_keys(decode_all_data, file_for_saves)

