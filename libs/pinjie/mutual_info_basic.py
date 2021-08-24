import numpy as np
from collections import Counter
from sklearn import metrics as mr
import time
import stitching

img_in_path = r'E:\compose\20201103\1-af\1-af\1-af 0.tif'


def entropy(labels):
    prob_dict = Counter(labels)
    s = sum(prob_dict.values())
    probs = np.array([i / s for i in prob_dict.values()])
    return - probs.dot(np.log(probs))


def random_list(num, img_min, img_max):
    randomlist = np.random.randint(img_min, img_max, size=num)
    return randomlist


def sig(label1, label2):
    sig_label = ["%s%s" % (i, j) for i, j in zip(label1, label2)]
    return sig_label


def basic(img, img_min, img_max):
    mis_basic = []
    x, y = img.shape[1:3]
    for i in range(200):  # 取一百个数，多的去掉
        i = (i + 1) * x
        list1 = random_list(i, img_min, img_max)
        list2 = random_list(i, img_min, img_max)
        list3 = random_list(i, img_min, img_max)
        list4 = random_list(i, img_min, img_max)
        list5 = random_list(i, img_min, img_max)
        list6 = random_list(i, img_min, img_max)
        list7 = random_list(i, img_min, img_max)
        list8 = random_list(i, img_min, img_max)
        list9 = random_list(i, img_min, img_max)
        list10 = random_list(i, img_min, img_max)

        mis_basic.append(((mr.mutual_info_score(list1, list2)) +
                          (mr.mutual_info_score(list3, list4)) +
                          (mr.mutual_info_score(list5, list6)) +
                          (mr.mutual_info_score(list7, list8)) +
                          (mr.mutual_info_score(list9, list10))) / 5)

    return mis_basic


def main():
    img = stitching.read_image(img_in_path)
    print(basic(img, 100, 500))

    return


if __name__ == '__main__':
    main()
