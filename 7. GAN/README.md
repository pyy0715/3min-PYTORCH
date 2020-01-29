# 학습내용 및 훈련결과 정리


**`Generative Adversarial Network`**
- [x] Generative: 생성 
- [x] Adversial: 적대적
- [x] Network: 신경망

**서로 적대적인 즉, 대립적인 관계에 있는 두 모델이 경쟁하면서 학습하여 무엇인가를 생성**

## gan.py

### Training Result
```{.python}
Epoch:[279/300],
D_loss_total: 1.1322,
G_loss: 0.9716,
D(x):0.72,
D(G(z)): 0.45
```

### model
**Inference**
```{.python}
model = Generator()
checkpoint = torch.load('./checkpoint/generator.pth.tar')
model.load_state_dict(checkpoint['Generator_state_dict'])
model
```

### checkpoint
**Define model and optimizer**
```{.python}
G = Generator().to(device)
D = Discriminator().to(device)
G_optimizer = optim.Adam(G.parameters(), lr=opt.lr)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr)
```

**load model and optimizer**
```
checkpoint1 = torch.load('./checkpoint/generator.pth.tar')
checkpoint2 = torch.load('./checkpoint/discriminator.pth.tar')

G.load_state_dict(checkpoint1['Generator_state_dict'])
G_optimizer_state_dict(checkpoint1['Generator_optimizer_state_dict'])

D.load_state_dict(checkpoint2['Discriminator_state_dict'])
D_optimizer_state_dict(checkpoint2['Discriminator_optimizer_state_dict'])
```
