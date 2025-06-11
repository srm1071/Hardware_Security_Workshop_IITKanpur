
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Thapar CPA AES Attack/aes_data_500k.csv', sep=',',index_col=None, header=None)
# We have 500k traces but we will only use 10k traces
# First and second columns of the traces have plaintexts and ciphertexts respectively
# There are total 282 columns
n = 10000
plaintexts = [str(df[0][i]) for i in range(n)]
ciphertexts = [str(df[1][i]) for i in range(n)]


traces = []
for i in range(279):
    traces.append(list(df[i+2]))
aes_sbox = [
    [int('63', 16), int('7c', 16), int('77', 16), int('7b', 16), int('f2', 16), int('6b', 16), int('6f', 16), int('c5', 16), int('30', 16), int('01', 16), int('67', 16), int('2b', 16), int('fe', 16), int('d7', 16), int('ab', 16), int('76', 16)],
    [int('ca', 16), int('82', 16), int('c9', 16), int('7d', 16), int('fa', 16), int('59', 16), int('47', 16), int('f0', 16), int('ad', 16), int('d4', 16), int('a2', 16), int('af', 16), int('9c', 16), int('a4', 16), int('72', 16), int('c0', 16)],
    [int('b7', 16), int('fd', 16), int('93', 16), int('26', 16), int('36', 16), int('3f', 16), int('f7', 16), int('cc', 16), int('34', 16), int('a5', 16), int('e5', 16), int('f1', 16), int('71', 16), int('d8', 16), int('31', 16), int('15', 16)],
    [int('04', 16), int('c7', 16), int('23', 16), int('c3', 16), int('18', 16), int('96', 16), int('05', 16), int('9a', 16), int('07', 16), int('12', 16), int('80', 16), int('e2', 16), int('eb', 16), int('27', 16), int('b2', 16), int('75', 16)],
    [int('09', 16), int('83', 16), int('2c', 16), int('1a', 16), int('1b', 16), int('6e', 16), int('5a', 16), int('a0', 16), int('52', 16), int('3b', 16), int('d6', 16), int('b3', 16), int('29', 16), int('e3', 16), int('2f', 16), int('84', 16)],
    [int('53', 16), int('d1', 16), int('00', 16), int('ed', 16), int('20', 16), int('fc', 16), int('b1', 16), int('5b', 16), int('6a', 16), int('cb', 16), int('be', 16), int('39', 16), int('4a', 16), int('4c', 16), int('58', 16), int('cf', 16)],
    [int('d0', 16), int('ef', 16), int('aa', 16), int('fb', 16), int('43', 16), int('4d', 16), int('33', 16), int('85', 16), int('45', 16), int('f9', 16), int('02', 16), int('7f', 16), int('50', 16), int('3c', 16), int('9f', 16), int('a8', 16)],
    [int('51', 16), int('a3', 16), int('40', 16), int('8f', 16), int('92', 16), int('9d', 16), int('38', 16), int('f5', 16), int('bc', 16), int('b6', 16), int('da', 16), int('21', 16), int('10', 16), int('ff', 16), int('f3', 16), int('d2', 16)],
    [int('cd', 16), int('0c', 16), int('13', 16), int('ec', 16), int('5f', 16), int('97', 16), int('44', 16), int('17', 16), int('c4', 16), int('a7', 16), int('7e', 16), int('3d', 16), int('64', 16), int('5d', 16), int('19', 16), int('73', 16)],
    [int('60', 16), int('81', 16), int('4f', 16), int('dc', 16), int('22', 16), int('2a', 16), int('90', 16), int('88', 16), int('46', 16), int('ee', 16), int('b8', 16), int('14', 16), int('de', 16), int('5e', 16), int('0b', 16), int('db', 16)],
    [int('e0', 16), int('32', 16), int('3a', 16), int('0a', 16), int('49', 16), int('06', 16), int('24', 16), int('5c', 16), int('c2', 16), int('d3', 16), int('ac', 16), int('62', 16), int('91', 16), int('95', 16), int('e4', 16), int('79', 16)],
    [int('e7', 16), int('c8', 16), int('37', 16), int('6d', 16), int('8d', 16), int('d5', 16), int('4e', 16), int('a9', 16), int('6c', 16), int('56', 16), int('f4', 16), int('ea', 16), int('65', 16), int('7a', 16), int('ae', 16), int('08', 16)],
    [int('ba', 16), int('78', 16), int('25', 16), int('2e', 16), int('1c', 16), int('a6', 16), int('b4', 16), int('c6', 16), int('e8', 16), int('dd', 16), int('74', 16), int('1f', 16), int('4b', 16), int('bd', 16), int('8b', 16), int('8a', 16)],
    [int('70', 16), int('3e', 16), int('b5', 16), int('66', 16), int('48', 16), int('03', 16), int('f6', 16), int('0e', 16), int('61', 16), int('35', 16), int('57', 16), int('b9', 16), int('86', 16), int('c1', 16), int('1d', 16), int('9e', 16)],
    [int('e1', 16), int('f8', 16), int('98', 16), int('11', 16), int('69', 16), int('d9', 16), int('8e', 16), int('94', 16), int('9b', 16), int('1e', 16), int('87', 16), int('e9', 16), int('ce', 16), int('55', 16), int('28', 16), int('df', 16)],
    [int('8c', 16), int('a1', 16), int('89', 16), int('0d', 16), int('bf', 16), int('e6', 16), int('42', 16), int('68', 16), int('41', 16), int('99', 16), int('2d', 16), int('0f', 16), int('b0', 16), int('54', 16), int('bb', 16), int('16', 16)]
]

reverse_aes_sbox = [
    [int('52', 16), int('09', 16), int('6a', 16), int('d5', 16), int('30', 16), int('36', 16), int('a5', 16), int('38', 16), int('bf', 16), int('40', 16), int('a3', 16), int('9e', 16), int('81', 16), int('f3', 16), int('d7', 16), int('fb', 16)],
    [int('7c', 16), int('e3', 16), int('39', 16), int('82', 16), int('9b', 16), int('2f', 16), int('ff', 16), int('87', 16), int('34', 16), int('8e', 16), int('43', 16), int('44', 16), int('c4', 16), int('de', 16), int('e9', 16), int('cb', 16)],
    [int('54', 16), int('7b', 16), int('94', 16), int('32', 16), int('a6', 16), int('c2', 16), int('23', 16), int('3d', 16), int('ee', 16), int('4c', 16), int('95', 16), int('0b', 16), int('42', 16), int('fa', 16), int('c3', 16), int('4e', 16)],
    [int('08', 16), int('2e', 16), int('a1', 16), int('66', 16), int('28', 16), int('d9', 16), int('24', 16), int('b2', 16), int('76', 16), int('5b', 16), int('a2', 16), int('49', 16), int('6d', 16), int('8b', 16), int('d1', 16), int('25', 16)],
    [int('72', 16), int('f8', 16), int('f6', 16), int('64', 16), int('86', 16), int('68', 16), int('98', 16), int('16', 16), int('d4', 16), int('a4', 16), int('5c', 16), int('cc', 16), int('5d', 16), int('65', 16), int('b6', 16), int('92', 16)],
    [int('6c', 16), int('70', 16), int('48', 16), int('50', 16), int('fd', 16), int('ed', 16), int('b9', 16), int('da', 16), int('5e', 16), int('15', 16), int('46', 16), int('57', 16), int('a7', 16), int('8d', 16), int('9d', 16), int('84', 16)],
    [int('90', 16), int('d8', 16), int('ab', 16), int('00', 16), int('8c', 16), int('bc', 16), int('d3', 16), int('0a', 16), int('f7', 16), int('e4', 16), int('58', 16), int('05', 16), int('b8', 16), int('b3', 16), int('45', 16), int('06', 16)],
    [int('d0', 16), int('2c', 16), int('1e', 16), int('8f', 16), int('ca', 16), int('3f', 16), int('0f', 16), int('02', 16), int('c1', 16), int('af', 16), int('bd', 16), int('03', 16), int('01', 16), int('13', 16), int('8a', 16), int('6b', 16)],
    [int('3a', 16), int('91', 16), int('11', 16), int('41', 16), int('4f', 16), int('67', 16), int('dc', 16), int('ea', 16), int('97', 16), int('f2', 16), int('cf', 16), int('ce', 16), int('f0', 16), int('b4', 16), int('e6', 16), int('73', 16)],
    [int('96', 16), int('ac', 16), int('74', 16), int('22', 16), int('e7', 16), int('ad', 16), int('35', 16), int('85', 16), int('e2', 16), int('f9', 16), int('37', 16), int('e8', 16), int('1c', 16), int('75', 16), int('df', 16), int('6e', 16)],
    [int('47', 16), int('f1', 16), int('1a', 16), int('71', 16), int('1d', 16), int('29', 16), int('c5', 16), int('89', 16), int('6f', 16), int('b7', 16), int('62', 16), int('0e', 16), int('aa', 16), int('18', 16), int('be', 16), int('1b', 16)],
    [int('fc', 16), int('56', 16), int('3e', 16), int('4b', 16), int('c6', 16), int('d2', 16), int('79', 16), int('20', 16), int('9a', 16), int('db', 16), int('c0', 16), int('fe', 16), int('78', 16), int('cd', 16), int('5a', 16), int('f4', 16)],
    [int('1f', 16), int('dd', 16), int('a8', 16), int('33', 16), int('88', 16), int('07', 16), int('c7', 16), int('31', 16), int('b1', 16), int('12', 16), int('10', 16), int('59', 16), int('27', 16), int('80', 16), int('ec', 16), int('5f', 16)],
    [int('60', 16), int('51', 16), int('7f', 16), int('a9', 16), int('19', 16), int('b5', 16), int('4a', 16), int('0d', 16), int('2d', 16), int('e5', 16), int('7a', 16), int('9f', 16), int('93', 16), int('c9', 16), int('9c', 16), int('ef', 16)],
    [int('a0', 16), int('e0', 16), int('3b', 16), int('4d', 16), int('ae', 16), int('2a', 16), int('f5', 16), int('b0', 16), int('c8', 16), int('eb', 16), int('bb', 16), int('3c', 16), int('83', 16), int('53', 16), int('99', 16), int('61', 16)],
    [int('17', 16), int('2b', 16), int('04', 16), int('7e', 16), int('ba', 16), int('77', 16), int('d6', 16), int('26', 16), int('e1', 16), int('69', 16), int('14', 16), int('63', 16), int('55', 16), int('21', 16), int('0c', 16), int('7d', 16)]
]

inv_Shift_row_lut = [0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]


def sbox_lookup(byte):
    x = byte >> 4
    y = byte & 15
    return aes_sbox[x][y]

def reverse_lookup(byte):
    x = byte >> 4
    y = byte & 15
    return reverse_aes_sbox[x][y]

def inv_shift_row(s):
    inv_s = ''
    inv_s += s[0:2] + s[26:28] + s[20:22] + s[14:16]
    inv_s += s[8:10] + s[2:4] + s[28:30] + s[22:24]
    inv_s += s[16:18] + s[10:12] + s[4:6] + s[30:32]
    inv_s += s[24:26] + s[18:20] + s[12:14] + s[6:8]

    return inv_s
import numpy as np

def custom_corrcoef(x, y):
    """
    Computes the Pearson correlation coefficient matrix between two 1D arrays.

    Args:
        x (np.ndarray): 1D array
        y (np.ndarray): 1D array

    Returns:
        np.ndarray: 2x2 correlation coefficient matrix
    """
    x = np.asarray(x)
    y = np.asarray(y)

    #Write your code here


def p_correlation(key_byte, byte_index, traces, plaintexts, ciphertexts):

    power_model = []

    for i, ciphertext in enumerate(ciphertexts):
        ind = inv_Shift_row_lut[byte_index]
        # cip_byte = int(ciphertext[2*ind : 2*(ind+1)], 16)
        addkey_outp = int(ciphertext[2*byte_index : 2*(byte_index+1)], 16) ^ key_byte
        sbox_outp = reverse_lookup(addkey_outp)
        power_model.append((sbox_outp^int(ciphertext[2*ind : 2*(ind+1)], 16)).bit_count())

    correlations = []
    for i in range(len(traces)):
        measurements = traces[i]
        x = [power_model, measurements]
        coefficient = custom_corrcoef(np.array(power_model), np.array(measurements))
        # coefficient = np.corrcoef(np.array(power_model), np.array(measurements))
        # coefficient = pearsonr(power_model, measurements)
        correlations.append(abs(coefficient[0][1]))

    return max(correlations), correlations
def get_correct_key_byte(byte_index, traces, plaintexts, ciphertexts):
    max_p = 0
    best_key_guess = 0
    p_list = []
    for key_guess in range(256):
        if key_guess%10==0:
            print(key_guess, end=' ')
        (p, temp) = p_correlation(key_guess, byte_index, traces, plaintexts, ciphertexts)
        p_list.append(temp)

        if p > max_p:
            max_p = p
            best_key_guess = key_guess

    return best_key_guess, max_p, p_list
def plot_corr(corr_mat, retrieved_key, p_title, correct_byte):
    plt.figure(figsize=(24,15))
    plt.title(p_title)
    for plot in range(16):
        plt.subplot(4,4,plot+1)
        plt.title("Byte " + str(plot))
        for i in range(256):
            col = 'blue'
            if i == retrieved_key[plot]:
                col = 'red'
            plt.plot(corr_mat[plot][i], color=col)
        plt.plot(corr_mat[plot][retrieved_key[plot]], color = 'orange')
        plt.plot(corr_mat[plot][int(correct_byte[2*plot : 2*(plot+1)], 16)], color = 'green')
        # plt.plot(corr_mat[plot][int(fake_byte[2*plot : 2*(plot+1)], 16)], color = 'red')

    plt.savefig(p_title + ".png")
    # plt.close()
final_corr_mat = []
for n in range(10000,10001,1):
    retrieved_key = []

    corr_mat = []
    sample_traces = [j[:n] for j in traces]
    for i in range(16):
        # print("The byte no: ", i)
        (best_key, max_p, p_list) = get_correct_key_byte(i, sample_traces, plaintexts[:n], ciphertexts[:n])
        retrieved_key.append(best_key)
        corr_mat.append(p_list)
        print("\nThe byte no: ", i, "is ", "{:02x}".format(best_key))
    print("The retrieved key using is", n, "samples: ", end='')

    subkey = ""
    for i in retrieved_key:
        subkey += "{:02x}".format(i)
        print("{:02x}".format(i), end = '')

    plot_corr(corr_mat, retrieved_key, "aes_bram2_100000", "d014f9a8c9ee2589e13f0cc8b6630ca6")

    final_corr_mat.append(corr_mat)

    # if subkey == "d014f9a8c9ee2589e13f0cc8b6630ca6":
    #     break

    print('')
