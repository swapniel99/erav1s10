from torch import optim
from torch.nn import functional as F
from matplotlib import pyplot as plt
from collections import defaultdict

from utils import get_device
from backprop import Train, Test


class Experiment(object):
    def __init__(self, model, dataset, lr=0.01, criterion=F.nll_loss):
        self.model = model.to(get_device())
        self.dataset = dataset
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, verbose=True, factor=0.1)
        self.train = Train(self.model, dataset, criterion, self.optimizer)
        self.test = Test(self.model, dataset, criterion)
        self.incorrect_preds = None

    def execute(self, epochs=20, target=None):
        target_count = 0
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}')
            self.train()
            test_loss, test_acc = self.test()
            if target is not None and test_acc >= target:
                target_count += 1
                if target_count >= 3:
                    print("Target Validation accuracy achieved thrice. Stopping Training.")
                    break
            self.scheduler.step(test_loss)

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
