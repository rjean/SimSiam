import torch
import glob
import os 
from PIL import Image, ImageOps
import random
import re
from natsort import natsorted
from deco import concurrent, synchronized
import numpy as np

@concurrent
def natsorted_p(data):
    return natsorted(data)

@synchronized
def natsorted_dict(dictionnary):
    for key in dictionnary:
        dictionnary[key] = natsorted_p(dictionnary[key])
    return dictionnary


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
    def __init__(self, root="datasets/objectron_96x96", split="train", memory=False, single=False, transform=None, 
                 debug_subset_size=None, return_indices = False, objectron_pair="uniform", objectron_exclude=[],
                 enable_cache=False, horizontal_flip=True):
        self.root=root
        self.memory = memory
        self.pairing = objectron_pair #Pairing strategy: uniform, next
        
        if "next_" in self.pairing or "previous_" in self.pairing:
            self.offsets = list(range(1,1+int(self.pairing.split("_")[1])))
            self.pairing = self.pairing.split("_")[0]
        print(f"Pairing mode: {self.pairing}. Possible offsets={self.offsets} Memory dataloader: {self.memory}")
        self.split = split
        self.transform = transform
        self.size = None
        self.single = single
        self.return_indices = return_indices
        self.enable_cache = enable_cache
        
        if "OBJECTRON_CACHE" in os.environ:
            self.enable_cache=True
        self.horizontal_flip = horizontal_flip
        #splits = glob.glob(f"{root}/*/")
        #if len(splits)==0:
        #    raise ValueError(f"Could not find splits in {root}")
        #splits = [x.split("/")[-2] for x in splits]
        #split = splits[0]
        self.categories = glob.glob(f"{root}/{split}/*/")
        self.categories = [x.split("/")[-2] for x in self.categories]
        self.categories.sort() #To have the same order as in the ImageFolder dataset.
        for exluded in objectron_exclude:
            self.categories.remove(exluded)
            print(f"Excluding {exluded} from dataset.")
        self.classes = self.categories

        #self.categories_list = []
        #for category in self.categories:
        #    self.categories_list.append(category)

        self.number_of_pictures = 0
        #
        #self.
        self.sequences_by_categories = {}
        for category in self.categories:
            self.sequences_by_categories[category] = self._get_sequences(category, self.split)

        self._load_samples(self.split)

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
            sequence_id = basename.split(".")[0] #"_".join([basename.split("_")[-3],basename.split("_")[-2],basename.split("_")[-1]])
            if sequence_id in sequences:
                sequences[sequence_id].append(basename)
            else:
                sequences[sequence_id] = [basename]

        sequences = natsorted_dict(sequences)
        #sequences = natsorted_dict(sequences)
        #for sequence_id in sequences: #Sort sequences
        #    sequences[sequence_id] = natsorted(sequences[sequence_id]) 
        return sequences

    def _load_samples(self, split="train", debug=True):
        samples = []
        for category in self.categories:
            self.number_of_pictures = 0
            for sequence in self.sequences_by_categories[category]:
                if len(self.sequences_by_categories[category][sequence])>5:
                    self.number_of_pictures+=len(self.sequences_by_categories[category][sequence])
                    for basename in self.sequences_by_categories[category][sequence]:
                        #frame_id = basename.split(".")[-2].split("_")[-1]
                        m = re.search("batch-(\d+)_(\d+)_(\d+).(\d+)\.jpg", basename)
                        batch_number = int(m[1])
                        sequence_number = int(m[2])
                        object_id = int(m[3])
                        frame_id = int(m[4])
                        sample = {"category": category, "sequence": sequence, 
                                  "basename": basename, "split": split, "frame_id": frame_id}
                        samples.append(sample)
                else:
                    print(f"Skipping {category}/{sequence} : Not enough samples!")
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
            dist = random.sample(self.offsets,1)[0] #For next or previous pairing modes.
            if self.pairing=="uniform":
                other_basename = random.sample(self.sequences_by_categories[category][sequence], 1)[0]
            elif self.pairing=="next":
                current_index = self.sequences_by_categories[category][sequence].index(basename)
                other_basename = self.get_next_basename(current_index, category, sequence, dist)
            elif self.pairing=="previous":
                current_index = self.sequences_by_categories[category][sequence].index(basename)
                other_basename = self.get_previous_basename(current_index, category, sequence, dist)
            elif self.pairing=="next_and_previous":
                current_index = self.sequences_by_categories[category][sequence].index(basename)
                if random.choice([True,False]):
                    other_basename = self.get_next_basename(current_index, category, sequence, dist)
                else:
                    other_basename = self.get_previous_basename(current_index, category, sequence, dist)
            elif self.pairing=="same":
                other_basename=basename #Basic SimSiam setup
            else:
                raise ValueError(f"Unsupported pairing scheme: {self.pairing}")
                #x=x+1
                #other_basename = self.sequences_by_categories[category][sequence]
            if other_basename!=basename or self.pairing=="same":
                image_path2 = f"{root}/{split}/{category}/{other_basename}"
                return image_path1, image_path2, category
        
        raise ValueError(f"Unable to find another different image for this batch. Please check if there is more than one sample in the sequence! {image_path1}")

    def get_next_basename(self, current_index, category, sequence, distance=1):
        assert distance < 5,f"Weird choice of maximum distance: {distance} ;). Comment out if this is intended"
        assert distance > 0,"Negative distances are invalid!"
        if (current_index+distance) < len(self.sequences_by_categories[category][sequence]):
            next_index=current_index+distance
        else:
            next_index=current_index-distance #For the last picture, give the previous one.
        next_basename = self.sequences_by_categories[category][sequence][next_index]
        return next_basename
    
    def get_previous_basename(self, current_index, category, sequence, distance=1):
        assert distance < 5,f"Weird choice of maximum distance: {distance} ;). Comment out if this is intended"
        assert distance > 0,"Negative distances are invalid!"

        if (current_index-distance)<0:
            previous_index=distance #For the first picture, give the next one instead of the previous one.
        else:
            previous_index=current_index-distance 
        next_basename = self.sequences_by_categories[category][sequence][previous_index]
        return next_basename
        
    def get_sequence_uid(self, idx):
        return self.samples[idx][""]

    def __getitem__(self, idx):
        success=False
        filename1 = None
        filename2 = None
        for i in range(0,5):
            #Some images are not having the right dimensions. We will simply skip them, and try the next one.
            filename1, filename2, category = self.get_pair_of_filenames(self.samples[idx+i], self.root)
            if filename1==filename2 and self.pairing!="same":
                continue #Sometimes, randomly sampling will give back the same file twice.
            
            if self.enable_cache:
                cache_filename1=f"{filename1}.npy"
                cache_filename2=f"{filename2}.npy"
                if not os.path.exists(cache_filename1):
                    image1 =Image.open(filename1)
                    image1= expand2square(image1)
                    np.save(cache_filename1, np.asarray(image1))
                if not os.path.exists(cache_filename2):
                    image2 =Image.open(filename2)
                    image2 = expand2square(image2)
                    np.save(cache_filename2, np.asarray(image2))
                cached_image1=np.load(cache_filename1)
                image1 = Image.fromarray(np.uint8(cached_image1))
                cached_image2=np.load(cache_filename2)
                image2 = Image.fromarray(np.uint8(cached_image2))
                
                
            else:
                image1 =Image.open(filename1)
                image2 =Image.open(filename2)
                image1 = expand2square(image1)
                image2 = expand2square(image2)

            if self.horizontal_flip and random.choice([True,False]) and not self.memory:
                image1 = ImageOps.mirror(image1)
                image2 = ImageOps.mirror(image2)
            
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

        uid = self.samples[idx]["category"] + "-" + self.samples[idx]["sequence"] + "-" + str(self.samples[idx]["frame_id"])
        meta = (torch.tensor(self.categories.index(category)), uid)
        if not self.single:
            return (image1, image2), meta
        else:
            return image1, meta

    def __len__(self):
        return len(self.samples)
