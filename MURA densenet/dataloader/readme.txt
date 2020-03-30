pipeline 		产生MURA-v1.1/train(valid)_XR_studytype.csv
preprocess	从相应的.csv文件中读取image，正则化然后存成.npy文件(在dataloader/NPY_img/中)
data_generator	从/dataloader/NPY_img/和/NPY_lab/中读取.npy文件

另见ImageNet_pretrained_Autoencoder
./data/pre_mura_npydata	从.csv文件中读取image，没有正则化(0-1，/255)，存在./data/文件夹
                训练Autoencoder时的图片没有正则化；
	训练DenseNet时的图片需要正则化；