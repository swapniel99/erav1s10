from custom_resnet import Model

model = Model(dropout=0.05)
model.summary(input_size=(512, 3, 32, 32))
