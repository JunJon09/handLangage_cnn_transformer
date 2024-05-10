import os
import shutil

#LSA64のデータセットをトレーニング用とテスト用に分割
def DatasetSplit():
    folder_path = "../LSA64/all/"
    files_and_directories = os.listdir(folder_path)

# ファイル名のみを取得（ディレクトリを除外）
    file_names = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]
    sorted_file_names = sorted(file_names)
    class_count = -1 #クラス数
    number = 0
    shutil.copy("../LSA64/all/008_010_003.mp4", "../test/01.mp4")

    train_list = []
    test_list = []
    for i, file_name in enumerate(sorted_file_names):
        if i % 50 == 0:
            class_count += 1
            print(i, class_count)
            path = "../test/pre/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)
            path = "../test/test/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)
            path = "../test/train/" + str(class_count).zfill(3)
            os.makedirs(path, exist_ok=True)

        basis_copy_path = "../LSA64/all/" + file_name
        x = i % 50
        if 0 <= x and x <30: # pre
            next_path = "../test/pre/" + str(class_count).zfill(3) + "/" + file_name
        elif 30 <= x and x < 40: #train
            next_path = "../test/train/" + str(class_count).zfill(3) + "/" + file_name
        else:#test
            next_path = "../test/test/" + str(class_count).zfill(3) + "/"  + file_name
        
        shutil.copy(basis_copy_path, next_path)


DatasetSplit()