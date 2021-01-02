import torch
import glob
import os 
from PIL import Image
import random

#https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class ObjectronDataset(torch.utils.data.Dataset):
    def __init__(self, root="datasets/objectron_96x96", split="train", train=True, single=False, transform=None, debug_subset_size=None):
        self.root=root
        self.split = split
        self.transform = transform
        self.size = None
        self.single = single
        #splits = glob.glob(f"{root}/*/")
        #if len(splits)==0:
        #    raise ValueError(f"Could not find splits in {root}")
        #splits = [x.split("/")[-2] for x in splits]
        #split = splits[0]
        self.categories = glob.glob(f"{root}/{split}/*/")
        self.categories = [x.split("/")[-2] for x in self.categories]
        self.categories.sort() #To have the same order as in the ImageFolder dataset.
        self.classes = self.categories

        #self.categories_list = []
        #for category in self.categories:
        #    self.categories_list.append(category)

        self.number_of_pictures = 0
        #
        self.sequences_by_categories = {}
        for category in self.categories:
            self.sequences_by_categories[category] = self._get_sequences(category, self.split)

        self._load_samples()
        if debug_subset_size is not None:
            self.samples = random.sample(self.samples, debug_subset_size)
        #categories        

    def _get_basenames(self, category, split="train"):
        files = glob.glob(f"{self.root}/{split}/{category}/*.jpg")
        basenames = [os.path.basename(x) for x in files]
        return basenames

    def _get_sequences(self, category, split="train"):
        basenames = self._get_basenames(category, split)
        sequences = {}
        for basename in basenames:
            sequence_id = "_".join([basename.split("_")[-3],basename.split("_")[-2]])
            if sequence_id in sequences:
                sequences[sequence_id].append(basename)
            else:
                sequences[sequence_id] = [basename]
        return sequences

    def _load_samples(self, split="train", debug=True):
        samples = []
        for category in self.categories:
            self.number_of_pictures = 0
            for sequence in self.sequences_by_categories[category]:
                if len(self.sequences_by_categories[category][sequence])>5:
                    self.number_of_pictures+=len(self.sequences_by_categories[category][sequence])
                    for basename in self.sequences_by_categories[category][sequence]:
                        sample = {"category": category, "sequence": sequence, "basename": basename, "split": split}
                        samples.append(sample)
                else:
                    print(f"Skipping {category}/{sequence} : Not enought samples!")
            print(f"Category {category} has {len(self.sequences_by_categories[category])} sequences, for a total of {self.number_of_pictures} pictures")
    
        print(f"Total of {len(samples)} samples")
        self.samples = samples

    def get_pair_of_filenames(self, sample, root):
        split = sample["split"]
        sequence = sample["sequence"]
        basename = sample["basename"]
        category = sample["category"]
        image_path1 = f"{root}/{split}/{category}/{basename}"
        for i in range(0,30):
            other_basename = random.sample(self.sequences_by_categories[category][sequence], 1)[0]
            if other_basename!=basename:
                image_path2 = f"{root}/{split}/{category}/{other_basename}"
                return image_path1, image_path2, category
        
        raise ValueError(f"Unable to find another different image for this batch. Please check if there is more than one sample in the sequence! {image_path1}")
        
        

    def __getitem__(self, idx):
        success=False
        filename1 = None
        filename2 = None
        for i in range(0,5):
            #Some images are not having the right dimensions. We will simply skip them, and try the next one.
            filename1, filename2, category = self.get_pair_of_filenames(self.samples[idx+i], self.root)
            if filename1==filename2:
                continue #Sometimes, randomly sampling will give back the same file twice.
            image1 =Image.open(filename1)
            image2 =Image.open(filename2)
            image1 = expand2square(image1)
            image2 = expand2square(image2)
            if image1.size != image2.size:
                print(f"Images not of the same size: {filename1}, {filename2}, skipping!")
                continue
            else:
                success=True
                if self.size == None:
                    self.size=image1.size
                if image1.size!=self.size or image2.size!=self.size:
                    print(f"Images not of the same size as previous images: {filename1}, {filename2}, skipping!")
                    continue
#                    raise ValueError(f"Images size not the same as previous ones: {filename1}, {filename2}!")
                break
        if not success:
            raise ValueError(f"Multiple images not having the right dimensions! {filename1}, {filename2}")
            
        if self.transform:
            image1, image2 = self.transform(image1), self.transform(image2)
        if not self.single:
            return (image1, image2), torch.tensor(self.categories.index(category))
        else:
            return image1, torch.tensor(self.categories.index(category))

    def __len__(self):
        return len(self.samples)
