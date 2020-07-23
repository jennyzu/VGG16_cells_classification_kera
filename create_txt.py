import xlrd
import numpy as np

workbook = xlrd.open_workbook('data_prep/donor_data.xlsx').sheet_by_index(0)
permut = np.random.permutation(workbook.nrows)
thres = int(0.7 * len(permut))
train_id = permut[0:thres]
val_id = permut[thres:]

#create train file
folder_name = []
folder_cls = []
dict_cls = {}
for i, row in enumerate(workbook):
    if i in train_id:
        folder_name.append(row[7])
        folder_cls.append(row[8])
        dict_cls[int(row[7])] = int(row[8])

for folder_i in range(len(train_id)):
    with open("train_file_wref_nocls0.txt", "a") as f1:
        if int(folder_cls[folder_i]) != 0:
            f1.write('/home/data_dl/ssl_24k_jenny/')
            f1.write(str(list(dict_cls.keys())[folder_i]))
            f1.write('/')
            f1.write('SEM_Defect_Internal')
            f1.write('.jpg')
            f1.write(' ')
            f1.write(str(list(dict_cls.values())[folder_i]))
            f1.write('\n')
            for j in range(1, 5):
                f1.write('/home/data_dl/ssl_24k_jenny/')
                f1.write(str(list(dict_cls.keys())[folder_i]))
                f1.write('/')
                f1.write('SEM_Defect_Topography ')
                f1.write(str(j))
                f1.write('.jpg')
                f1.write(' ')
                f1.write(str(list(dict_cls.values())[folder_i]))
                f1.write('\n')
            f1.write('/home/data_dl/ssl_24k_jenny/')
            f1.write(str(list(dict_cls.keys())[folder_i]))
            f1.write('/')
            f1.write('SEM_Reference_Internal')
            f1.write('.jpg')
            f1.write(' ')
            f1.write(str(list(dict_cls.values())[folder_i]))
            f1.write('\n')
            for j in range(1, 5):
                f1.write('/home/data_dl/ssl_24k_jenny/')
                f1.write(str(list(dict_cls.keys())[folder_i]))
                f1.write('/')
                f1.write('SEM_Reference_Topography ')
                f1.write(str(j))
                f1.write('.jpg')
                f1.write(' ')
                f1.write(str(list(dict_cls.values())[folder_i]))
                f1.write('\n')

'''
with open('defects_list_with0.csv') as csv_file:
csv_reader = csv.reader(csv_file, delimiter=',')
#create validation file
folder_name = []
folder_cls = []
dict_cls = {}
for i, row in enumerate(csv_reader):
    if i in val_id:
        folder_name.append(row[7])
        folder_cls.append(row[8])
        dict_cls[int(row[7])] = int(row[8])

for folder_i in range(len(val_id)):
    with open("validation_file_wref_nocls0.txt", "a") as f1:
        if int(folder_cls[folder_i]) != 0:
            f1.write('/home/data_dl/ssl_24k_jenny/')
            f1.write(str(list(dict_cls.keys())[folder_i]))
            f1.write('/')
            f1.write('SEM_Defect_Internal')
            f1.write('.jpg')
            f1.write(' ')
            f1.write(str(list(dict_cls.values())[folder_i]))
            f1.write('\n')
            for j in range(1, 5):
                f1.write('/home/data_dl/ssl_24k_jenny/')
                f1.write(str(list(dict_cls.keys())[folder_i]))
                f1.write('/')
                f1.write('SEM_Defect_Topography ')
                f1.write(str(j))
                f1.write('.jpg')
                f1.write(' ')
                f1.write(str(list(dict_cls.values())[folder_i]))
                f1.write('\n')
            f1.write('/home/data_dl/ssl_24k_jenny/')
            f1.write(str(list(dict_cls.keys())[folder_i]))
            f1.write('/')
            f1.write('SEM_Reference_Internal')
            f1.write('.jpg')
            f1.write(' ')
            f1.write(str(list(dict_cls.values())[folder_i]))
            f1.write('\n')
            for j in range(1, 5):
                f1.write('/home/data_dl/ssl_24k_jenny/')
                f1.write(str(list(dict_cls.keys())[folder_i]))
                f1.write('/')
                f1.write('SEM_Reference_Topography ')
                f1.write(str(j))
                f1.write('.jpg')
                f1.write(' ')
                f1.write(str(list(dict_cls.values())[folder_i]))
                f1.write('\n')
'''