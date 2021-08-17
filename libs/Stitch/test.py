from sympy import diff
from sympy import symbols
import math
import pandas as pd

A = [2.1887339E+00,-9.6930568E-03,9.1515480E-03,6.4656751E-05,9.1890058E-06,-4.7114927E-07]

def func(x):

    return (A[0] + (A[1] * x ** 2) + (A[2] * x ** (-2)) + (A[3] * x ** (-4)) + (A[4] * x ** (-6)) + (
                A[5] * x ** (-8))) ** 0.5

def compute_num(x, func_str):
    return eval(func_str)


def calculate_once():
    result_list = []
    w1 = 680e-9
    w2 = 1050e-9
    w3 = 1300e-9
    for wave_length in [w1, w2, w3]:
        x = symbols("x")
        d2n_dlbd2 = diff(func(x), x, 2)
        print(f'd2n_dlbd2 function = {str(d2n_dlbd2)}')

        C = 3e8
        tao_p = 120e-15
        result = 4e-2 * wave_length**3 * 0.315 / C / tao_p * compute_num(wave_length, str(d2n_dlbd2))
        # print(f'd2n_dlbd2 = {eval(str(d2n_dlbd2))}')
        # print(f'result = {result}')

        result_list.append(result)
    return result_list


def main():
    input_dict = {}
    result_dict = {}  # {'material name': [680 result, 1050 result, 1300 result]}

    # Read excel
    path = r"C:\Users\CXC\Desktop\cdgm201904.xls"
    data = pd.read_excel(path, sheet_name=0)
    # print(data)
    for indexs in data.index:
        # if indexs not in list(range(1, 247+1)):
        if indexs not in list(range(1, 10+1)):
             continue
        data_row = data.loc[indexs].values[21:27]
        material_name = data.loc[indexs].values[0]
        input_dict[str(material_name)] = data_row

    # Read Material and A0-A5
    from tqdm import tqdm
    for material_name, A_in in tqdm(input_dict.items()):
        global A
        A = A_in
        result_dict[material_name] = calculate_once()

    # Print result
    for material_name, result_list in result_dict.items():
        print(f'{material_name:10s} => {result_list}')


if __name__ == '__main__':
    main()