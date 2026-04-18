import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class Dermnet(DatasetBase):

    dataset_dir = "Dermnet"

    def __init__(self, cfg): # cfg
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)) # cfg.DATASET.ROOT
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir # os.path.join(self.dataset_dir, "Data")
        self.preprocessed = os.path.join("/data/yanjie/code/OOD/LoCoOp_ori/data/", os.path.basename(self.dataset_dir), "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join("/data/yanjie/code/OOD/LoCoOp_ori/data/", os.path.basename(self.dataset_dir), "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            classnames = self.read_classnames(os.path.join(self.image_dir, "train"))
            train = self.read_data(classnames, "train")
            test = self.read_data(classnames, "test")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS # cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED # cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES # cfg.DATASET.SUBSAMPLE_CLASSES
        ori_train = train
        ori_val = test
        ori_test = test
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)
        _, _, id = OxfordPets.subsample_classes(ori_train, ori_test, ori_test, subsample='base')
        _, _, ood = OxfordPets.subsample_classes(ori_train, ori_test, ori_test, subsample='new')
        self.id = id
        self.ood = ood

        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, classnames, split_dir):  # 按训练集验证集划分
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder), sort=True)
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_classnames(self, image_dir):   # 把文件夹对应成类名
        classnames = OrderedDict()
        crouse_classnames = os.listdir(image_dir)
        for i, classname in enumerate(crouse_classnames):
            classnames[classname] = classname
        return classnames
    
if __name__ == "__main__":
    pass