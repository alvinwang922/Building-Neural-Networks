import torch
import torch.nn as nn
import torch.optim as optim

#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 3

for epoch in range(EPOCHS):  # 3 full passes over the data
    for data in trainset:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        # sets gradients to 0 before loss calc. You will do this likely every step.
        net.zero_grad()
        # pass in the reshaped batch (recall they are 28x28 atm)
        output = net(X.view(-1, 784))
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!


correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        # print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

print(torch.argmax(net(X[0].view(-1, 784))[0]))
plt.imshow(X[0].view(28, 28))
plt.show()
