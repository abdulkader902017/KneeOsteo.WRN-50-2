import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets

from util import plot_confusion_matrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
torch.cuda.empty_cache()
    
def load_data():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = transforms.Resize((224, 224))

    preprocessor = transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.ImageFolder(r"C:\Users\abdul\Desktop\Research\Knee Osteo\56rmx5bjcr-1\train\train", preprocessor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

    testset = datasets.ImageFolder(r"C:\Users\abdul\Desktop\Research\Knee Osteo\56rmx5bjcr-1\train\test", preprocessor)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

    return trainloader, testloader


class WideResNet50(nn.Module):
    def __init__(self, nb_of_epochs, learning_rate):
        super(WideResNet50, self).__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.aux_logits = False # 
        num_ftrs = self.model.fc.in_features
        self.nb_classes = 3
        self.model.fc = nn.Linear(num_ftrs, self.nb_classes)
        self.nb_of_epochs = nb_of_epochs
        self.learning_rate = learning_rate

    def fit(self, trainloader):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # setup SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)

        for epoch in range(self.nb_of_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute forward pass
                outputs = self.model.forward(inputs)

                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data

            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i + 1)))
            running_loss = 0.0

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        test_labels = []

        for images, labels in testloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = self.model.forward(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            test_labels += labels.cpu().numpy().tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_predicted += predicted.cpu().numpy().tolist()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

        return test_labels, all_predicted


def main():
    # get data
    trainloader, testloader = load_data()

    model = WideResNet50(nb_of_epochs=17, learning_rate=0.0001).cuda()
    model.fit(trainloader) 
    test_labels, predicted_labels = model.predict(testloader)

    torch.save(model, "results/WideResNet50")

    plt.figure(1)
    plot_confusion_matrix(predicted_labels, test_labels, "Wide-ResNet-50-2")
    plt.savefig('results/conf_finetune.png')
    plt.show()


if __name__ == '__main__':
    main()
