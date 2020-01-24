# 학습내용 및 훈련결과 정리

## fgsm_attack.py

> FGSM(Fast Gradient Sign Method)은 반복된 학습 없이 잡음을
> 생성하는 원샷 공격으로, 입력 이미지에 대한 기울기의 정보를 
> 추출하여 잡음을 생성.

> **Data**: ./'imagnet_samples/corgie.jpg'

* 적대적 공격은 적절한 잡음을 생성해 사람의 눈에는 똑같이 보이지만, 모델을 헷갈리게 만드는 적대적 예제를 생성하는 것이 핵심.

**분류기준에 따른 잡음의 생성 방법**
1. 기울기와 같은 모델정보가 필요한지(White Box /  Black Box)
2. 원하는 정답으로 유도할 수 있는지(Targeted / Non-Targeted)
3. 반복된 학습(최적화)이 필요한지(Iterative / One-Shot)
4. 특정 입력에만 적용되는지 / 모든 이미지에 적용될수 있는지
