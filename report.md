# Implementacija rada: "An Abstract Domain for Certifying Neural Networks"

Veštačke neuronske mreže su u poslednjih 15-ak godina donele veliki napredak u najrazličitijim zadacima (računarska vizija, razumevanje prirodnog jezika, robotika...). Medjutim, neuronske mreže su i dalje crne kutije, što ograničava njihovu upotrebu u kontekstima u kojima greške mogu imati velike posledice. Štaviše, pokazano je da se u mnogim domenima neuronske mreže mogu "zavarati" tako što se napravi neprimetna promena u ulaznim podacima [1]. Ovakvi izmenjeni ulazi se zovu Adversarijalni primeri.


[1]: Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining and Harnessing Adversarial Examples