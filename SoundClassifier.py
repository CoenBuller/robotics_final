from torch import Tensor, nn

# This is the CNN architecture that we will use for sound classification. 
# It takes in a mel-spectrogram of shape (1, n_mfcc, n_frames) and outputs a probability distribution over the n_classes possible classes.
class AudioCNN(nn.Module):
    def __init__(self, n_classes: int, n_mfcc: int = 62):
        super().__init__()
        self.n_classes = n_classes
        self.n_mfcc    = n_mfcc
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Identity(),                                              
            nn.Linear(in_features=32, out_features=n_classes),          
            nn.Softmax(dim=-1),                                         
        )

# The forward method defines how the input tensor flows through the network layers to produce the output.
    def forward(self, x: Tensor):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)