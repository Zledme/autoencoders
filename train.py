from autoencoder import Autoencoders
import torchvision
from torchvision import transforms

def train(model, optimizer, loss_fn):

    batch_size = 32
    epochs = 20
    lr = 0.0005


    model.train(True)
    for epoch in range(epochs):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(training_loader[:500]):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs,inputs)
            loss.backward()

            optimizer.step()

            running_loss ++ loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        print(last_loss)
    return model

def train_autoencoder():
    # Define a transform to convert PIL images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)).
        transforms.Lambda(lambda x: torch.reshape(x, (1,)))
        ])
    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Download and load the test data
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    #init the model
    autoencoder = Autoencoders(
            input_shape=(1, 28, 28),
            conv_filters=(32, 64, 64, 64),
            conv_kernels=(3, 3, 3, 3),
            conv_strides=(1, 2, 2, 1),
            latent_space_dim=2
        )
    model = train(autoencoder, optimizer, loss_fn)
    return model




