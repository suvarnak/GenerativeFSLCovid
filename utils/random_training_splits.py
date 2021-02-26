import os
import shutil
import random



def copy_random_k_files(src_dir, k, dst_dir):
    file_list = os.listdir(src_dir)
    if k == -1:
        k=len(file_list)
    for i in range(k):
        random_file=random.choice(file_list)
        print(random_file)
        src1 = os.path.join(src_dir, random_file)
        dst1 = os.path.join(dst_dir, random_file)
        shutil.copyfile(src1, dst1)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def main():
    shots_per_run = 84
    no_of_runs =10
    image_dir = "./data/DeepCovid"
    split_names = os.listdir(image_dir)
    target_splits_dir = "./data"
    print("createing directory structure")

    for i in range(no_of_runs):
        random_run_path = os.path.join(target_splits_dir, "DeepCovid_"+str(shots_per_run) + "_" + str(i))
        print(random_run_path)
        os.mkdir(random_run_path)
        train_split = "train" #split_names[1] 
        test_split = "test" #split_names[0] 
        class_names = ['0_non','1_covid'] 
        base_path_split = os.path.join(random_run_path,train_split)
        os.makedirs(os.path.join(base_path_split,class_names[0])) 
        os.makedirs(os.path.join(base_path_split,class_names[1])) 
        base_path_split = os.path.join(random_run_path,test_split)
        os.makedirs(os.path.join(base_path_split,class_names[0])) 
        os.makedirs(os.path.join(base_path_split,class_names[1])) 
        print("Directory '% s' created" % random_run_path) 
        src_train_dir = os.path.join(image_dir,"train")
        src_train_dir_non = os.path.join(src_train_dir,"0_non")
        src_train_dir_covid = os.path.join(src_train_dir,"1_covid")
        dst_train_dir = os.path.join(random_run_path,"train")
        dst_train_dir_non = os.path.join(dst_train_dir,"0_non")
        dst_train_dir_covid = os.path.join(dst_train_dir,"1_covid")
        copy_random_k_files(src_train_dir_non, shots_per_run, dst_train_dir_non)
        copy_random_k_files(src_train_dir_covid, shots_per_run, dst_train_dir_covid)
        src_test_dir = os.path.join(image_dir,"test")
        src_test_dir_non = os.path.join(src_test_dir,"0_non")
        src_test_dir_covid = os.path.join(src_test_dir,"1_covid")
        dst_test_dir = os.path.join(random_run_path,"test")
        dst_test_dir_non = os.path.join(dst_test_dir,"0_non")
        dst_test_dir_covid = os.path.join(dst_test_dir,"1_covid")
        copytree(src_test_dir_non, dst_test_dir_non)
        copytree(src_test_dir_covid, dst_test_dir_covid)


if __name__ == '__main__':
    main()

