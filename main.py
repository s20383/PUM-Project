import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms


def define_image_par():
    custom_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(900),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(saturation=0.2),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ])
    return custom_transforms


def train_model(train_folder, test_folder, custom_transforms):
    Train_dataset = core.Dataset(train_folder, transform=custom_transforms)
    Test_dataset = core.Dataset(test_folder)
    loader = core.DataLoader(Train_dataset, batch_size=2, shuffle=True)
    model = core.Model(["tablica"])
    losses = model.fit(loader, Test_dataset, epochs=7,
                       lr_step_size=5, learning_rate=0.001, verbose=True)
    return loader, model, losses


def show_labeled_image(path, model):
    image = utils.read_image(path)
    labels, boxes, scores = model.predict(image)
    thresh = 0.5
    filtered_indices = np.where(scores > thresh)
    filtered_scores = scores[filtered_indices]
    filtered_boxes = boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    show_labeled_image(image, filtered_boxes, filtered_labels)


def main():
    custom_transforms = define_image_par()
    loader, model, losses = train_model(
        'Treningowy', 'Testowy', custom_transforms)
    joblib.dump(model, 'model.sav')
    plt.plot(losses)
    plt.show()
    show_labeled_image('renault-clio-iv-grandtour.jpg', model)
    show_labeled_image('Renault-Megane-Grandtour-11-scaled.jpg', model)
    show_labeled_image(
        'renault-megane-iv-2017-15-dci-grand-tour-nieuszkodzony-wielkopolskie-556375465.jpg', model)


if __name__ == '__main__':
    main()
