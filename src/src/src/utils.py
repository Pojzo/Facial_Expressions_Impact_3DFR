# Author: Peter Kovac

import os
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle


class obj2pcl: 
    @staticmethod
    def convert(path, num_points=999, ouput_name=None):
        if not os.path.exists(path):
            raise FileNotFoundError("The path does to the source .obj file does not exist", path)
        
        mesh = o3d.io.read_triangle_mesh(path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
        
        return point_cloud, np.asarray(point_cloud.points)

def rotate_point_cloud(point_cloud, rotation_matrix):
    return np.dot(point_cloud, torch.as_tensor(rotation_matrix))

def draw_curve(data, config, savepath=None):
    plt.title(config.get("title") + " " + config.get("num_classes") + "_classes" + config.get("num_points") + "_points")
    if (config.get("title") == "Accuracy"):
        plt.ylim([0, 101])
        # limit the data to 98% for better visualization
        # data = [x if x < 98 else 98 for x in data]

    plt.plot(data)
    plt.xlabel("Epoch")
    plt.ylabel(config.get("title"))

    print("saving to: ", savepath + "/" + config.get("title") + "_" + config.get("num_classes") + "_classes" + config.get("num_points") + "_points.png")
    print(savepath is not None)
    if savepath is not None:
        plt.savefig(savepath + "/" + config.get("title"))
        with open(savepath + "/" + config.get("title") + "_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        with open(savepath + "/" + config.get("title") + "_config.pkl", 'wb') as f:
            pickle.dump(config, f)
        plt.close()
    else:
        plt.show()
        plt.clf()



class CustomDataset(Dataset):
    def __init__(self, path_to_data, num_points=1024, num_classes=5, train_mode="neutral-neutral", train=True):
        if not os.path.exists(path_to_data):
            raise FileNotFoundError("The path does not exist")
        
        self.expressions = ['anger', 'confusion', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']        
        self.num_classes = num_classes
        self.num_points = 1024
        self.datadir = path_to_data
        self.train_mode = train_mode
        self.train = train
        self.absolute_path = os.path.join(os.getcwd(), self.datadir)
        # self.labels = self.find_classes()
        # self.data = self.create_data()
        
        # print("Found {} classes".format(self.__len__()))
        
        # Create class-to-index mapping
        # self.class_to_idx = {class_: idx for idx, class_ in enumerate(self.labels)}
            
    def find_classes(self):
        directories = os.listdir(self.datadir)
        self.classes =  [x for x in directories if os.path.isdir(os.path.join(self.absolute_path, x))][:self.num_classes]
        import torch
        
    def create_data(self):
        data = []
        self.idx_to_class = {}
        self.idx_to_expression = {}
        # first get the neutral face
        for class_ in self.classes:
            neutral_obj_path = os.path.join(self.absolute_path, class_, 'anger', class_, 'reconstruction', class_ +  '.obj')
            expression_paths = []
            for expression in self.expressions:
                expression_path = os.path.join(self.absolute_path, class_, expression, class_, 'animation', class_ +  '.obj')
                expression_paths.append(expression_path)
    
            _, neutral_pc_array = obj2pcl.convert(neutral_obj_path, num_points=self.num_points)
    
            data.append(neutral_pc_array)
            self.idx_to_class[len(data) - 1] = class_
            self.idx_to_expression[len(data) - 1] = "neutral"
    
            if self.train and not self.train_mode == "modified-modified":
                continue
                
            if not self.train and self.train_mode == "neutral-neutral":
                continue
               
            for expression, expression_path in zip(self.expressions, expression_paths):
                try:
                    _, pc_array = obj2pcl.convert(expression_path, num_points=self.num_points)
                except FileNotFoundError:
                    print("File not found for: ", expression_path)
                    print("This is none and should be")
                    pc_array = None
                    continue
    
                data.append(pc_array)
                self.idx_to_class[len(data) - 1] = class_
                self.idx_to_expression[len(data) - 1] = expression
    
        self.data = [torch.as_tensor(d) for d in data]

        for i in range(len(self.data)):
            new_data = self.transform_data(self.data[i])
            self.data.append(new_data)

            self.idx_to_class[len(self.data) - 1] = self.idx_to_class[i]
            self.idx_to_expression[len(self.data) - 1] = self.idx_to_expression[i]
        
    
    # written by chatgpt
    def rotate_point_cloud(self, point_cloud, rotation_angle):
        """Rotate the point cloud around the Y axis by a given angle."""
        rotation_matrix = np.array([
            [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
            [0, 1, 0],
            [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
        ])
        return np.dot(point_cloud, rotation_matrix)
    
    # written by chatgpt
    def translate_point_cloud(self, point_cloud, translation_vector):
        """Translate the point cloud by a given vector."""
        return point_cloud + translation_vector
    
    # written by chatgpt
    def scale_point_cloud(self, point_cloud, scale_factor):
        """Scale the point cloud by a given factor."""
        return point_cloud * scale_factor
    
    # written by chatgpt
    def jitter_point_cloud(self, point_cloud, sigma=0.01, clip=0.05):
        """Add Gaussian noise to the point cloud."""
        jittered_data = np.clip(sigma * np.random.randn(*point_cloud.shape), -1*clip, clip)
        return point_cloud + jittered_data
    
    # written by chatgpt
    def flip_point_cloud(self, point_cloud, axis):
        """Flip the point cloud across a given axis."""
        return point_cloud * np.array([-1 if i == axis else 1 for i in range(point_cloud.shape[1])])

    # written by chatgpt
    def transform_data(self, point_cloud):
        """Apply a random transformation to a point cloud."""
        transformations = [self.rotate_point_cloud, self.translate_point_cloud, self.scale_point_cloud, self.jitter_point_cloud, self.flip_point_cloud]
        transformation = np.random.choice(transformations)
        if transformation == rotate_point_cloud:
            angle = np.random.uniform(-np.pi, np.pi)
            return transformation(point_cloud, angle)
        elif transformation == self.translate_point_cloud:
            vector = np.random.uniform(-1, 1, size=3)
            return transformation(point_cloud, vector)
        elif transformation == self.scale_point_cloud:
            factor = np.random.uniform(0.5, 1.5)
            return transformation(point_cloud, factor)
        elif transformation == self.jitter_point_cloud:
            return transformation(point_cloud)
        elif transformation == self.flip_point_cloud:
            axis = np.random.choice([0, 1, 2])
            return transformation(point_cloud, axis) 


    def __getattr__(self, attribute_name):
        print("Attribute not found: ", attribute_name)
        return super().__getattr__(attribute_name)
    
    def is_neutral(self, idx):
        return self.idx_to_expression[idx] == "neutral"

    def get_expression(self, idx):
        return self.idx_to_expression[idx]

    def __len__(self):
        return len(self.data)
    
    def rotation_matrix_z(self, theta):
        theta = torch.tensor(theta).to(torch.float64)
        return torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ]).to(torch.float64)

    def __getitem__(self, index):
        return {'pos': self.data[index].type(torch.float32), 
            'x': self.data[index].type(torch.float32),
            'y': self.classes.index(self.idx_to_class[index]),
            'exp': self.get_expression(index),
        }
        
def create_datasets(dest, cfg):
    train_mode = cfg.train_mode
    if train_mode == "neutral-neutral":
        train_dataset = CustomDataset(dest, num_points=cfg.num_points, num_classes=cfg.num_classes, train_mode=train_mode, train=True)
        train_dataset.find_classes()
        train_dataset.create_data()

        test_dataset = train_dataset

        return train_dataset, test_dataset
        
    elif train_mode == "neutral-modified":
        train_dataset = CustomDataset(dest, num_points=cfg.num_points, num_classes=cfg.num_classes, train_mode=train_mode, train=True)
        train_dataset.find_classes()
        train_dataset.create_data()

        test_dataset = CustomDataset(dest, num_points=cfg.num_points, num_classes=cfg.num_classes, train_mode=train_mode, train=False)
        test_dataset.find_classes()
        test_dataset.create_data()

        return train_dataset, test_dataset
    
    elif train_mode == "modified-modified":
        train_dataset = CustomDataset(dest, num_points=cfg.num_points, num_classes=cfg.num_classes, train_mode=train_mode)
        train_dataset.find_classes()
        train_dataset.create_data()

        test_dataset = CustomDataset(dest, num_points=cfg.num_points, num_classes=cfg.num_classes, train_mode=train_mode)
        test_dataset.find_classes()
        test_dataset.create_data()

        return train_dataset, test_dataset
    
def rotate_z(points, angle):
    # create the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    # apply the rotation matrix to the points
    return np.dot(points, rotation_matrix)

def rotate_x(points, angle):
    # create the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])

    # apply the rotation matrix to the points
    return np.dot(points, rotation_matrix)
