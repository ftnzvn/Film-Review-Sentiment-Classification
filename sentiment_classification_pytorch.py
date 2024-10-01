import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.model_selection import train_test_split

# creating dummy sample data for film reviews
data = [
    ("I actually really enjoyed How I Met your Mother", 1),
    ("I loved Interstellar", 1),
    ("Whiplash was a cinematic masterpiece", 1),
    ("the Tearsmith is the worst movie I've ever seen", 0),
    ("I did not enjoy watching Love Actually", 0),
    ("the movie adaptation of Paper Towns was poor", 0),
    ("the acting in All of Us Strangers was phenomenal", 1),
    ("Inception was mind-blowing", 1),
    ("I loved the colour scheme in Blade Runner 2049", 1),
    ("the storyline of The Shawshank Redemption was so moving", 1),
    ("Parasite was brilliant from start to finish", 1),
    ("the characters in Toy Story are so lovable", 1),
    ("The Matrix is a timeless classic", 1),
    ("The cinematography in The Grand Budapest Hotel was fantastic", 1),
    ("Watching Spirited Away was a magical experience", 1),
    ("The direction of 12 Angry Men was flawless", 1),
    ("I thoroughly enjoyed Back to the Future", 1),
    ("Joker was an intense and gripping film", 1),
    ("I found The Room to be utterly terrible", 0),
    ("Cats was a complete mess", 0),
    ("Movie 43 was just awful", 0),
    ("I did not enjoy anything about Fifty Shades of Grey", 0),
    ("The acting in Twilight was cringe-worthy", 0),
    ("The jokes in Holmes & Watson were painfully unfunny", 0),
    ("Suicide Squad was a total letdown", 0),
    ("I couldn't stand the characters in The Emoji Movie", 0),
    ("The effects in Birdemic were hilariously bad", 0)
]

# randomly shuffle dataset
random.shuffle(data)

# splitting into training and validation sets
train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        tokens = sentence.lower().split()
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices), torch.tensor(label)

# building vocab
def yield_tokens(data_iter):
    for sentence, _ in data_iter:
        yield sentence.lower().split()

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

train_dataset = TextDataset(train_data, vocab)
valid_dataset = TextDataset(valid_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: list(zip(*x)))


# preprocessing text (padding to make same sentence length)
def collate_batch(batch):
    sentences, labels = batch
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=vocab['<unk>'])
    labels = torch.tensor(labels)
    return padded_sentences, labels


# building rnn to classify text
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  #embedding layer (converts word indices into vector representations)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)   #fully connected layer for classification

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        output = self.fc(h_n.squeeze(0))
        return output

# setting parameters
vocab_size = len(vocab)
embed_dim = 8
hidden_dim = 16
output_dim = 1

model = SimpleRNN(vocab_size, embed_dim, hidden_dim, output_dim)

#training rnn with cross-entropy loss and stochastic gradient descent

criterion = nn.BCEWithLogitsLoss()   #using binary cross entropy w logits as loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 20
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        sentences, labels = collate_batch(batch)
        optimizer.zero_grad()
        outputs = model(sentences).squeeze(1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

#model evaluationn
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            sentences, labels = collate_batch(batch)
            outputs = model(sentences).squeeze(1)
            predictions = torch.round(torch.sigmoid(outputs))
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Validation accuracy
accuracy = evaluate(model, valid_loader)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
