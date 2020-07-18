import numpy as np
import torch

NUM_DIGITS = 10
NUM_HIDDEN = 100
BATCH_SIZE = 256

def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])[::-1]

trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)

if torch.cuda.is_available():
    model = model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(3000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX) # forward
        loss = loss_fn(y_pred, batchY)

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward() # backward pass
        optimizer.step() # gradient pass

# testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
# if torch.cuda.is_available():
#     testX = testX.cuda()
# with torch.no_grad():
#     testY = model(testX)
#
# prediction = zip(range(1, 101), testY.max(1)[1].cpu().data.tolist())
# print([fizz_buzz_encode(i) for i, x in prediction])