from torch import nn, optim
from matplotlib import pyplot as plt
from collections import defaultdict
from torch_lr_finder import LRFinder

from utils import get_device
from backprop import Train, Test


class Experiment(object):
    def __init__(self, model, dataset, criterion=None, epochs=24):
        self.device = get_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=1e-2)
        self.best_lr = self.find_lr()
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.best_lr,
            steps_per_epoch=len(self.dataset.train_loader),
            epochs=self.epochs,
            pct_start=5/self.epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        self.train = Train(self.model, dataset, self.criterion, self.optimizer)
        self.test = Test(self.model, dataset, self.criterion)
        self.incorrect_preds = None

    def find_lr(self):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
        lr_finder.range_test(self.dataset.train_loader, end_lr=1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
        return best_lr

    def execute(self, target=None):
        target_count = 0
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}, LR {self.scheduler.get_last_lr()}')
            self.train()
            test_loss, test_acc = self.test()
            if target is not None and test_acc >= target:
                target_count += 1
                if target_count >= 3:
                    print("Target Validation accuracy achieved thrice. Stopping Training.")
                    break
            self.scheduler.step()

    def show_incorrect(self, denorm=True):
        self.incorrect_preds = defaultdict(list)
        self.test(self.incorrect_preds)

        _ = plt.figure(figsize=(10, 3))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.tight_layout()
            image = self.incorrect_preds["images"][i].cpu()
            if denorm:
                image = self.dataset.denormalise(image)
            plt.imshow(self.dataset.show_transform(image), cmap='gray')
            pred = self.incorrect_preds["predicted_vals"][i]
            truth = self.incorrect_preds["ground_truths"][i]
            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            plt.title(f'{pred}/{truth}')
            plt.xticks([])
            plt.yticks([])
