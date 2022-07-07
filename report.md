# Implementacija rada: "An Abstract Domain for Certifying Neural Networks"

## Uvod

Veštačke neuronske mreže su u poslednjih 15-ak godina donele veliki napredak u najrazličitijim zadacima (računarska vizija, razumevanje prirodnog jezika, robotika...). Medjutim, neuronske mreže su i dalje crne kutije, što ograničava njihovu upotrebu u kontekstima u kojima greške nisu dopustive. Štaviše, pokazano je da se u mnogim domenima neuronske mreže mogu "zavarati" tako što se napravi neprimetna promena u ulaznim podacima [1]. Ovako izmenjeni ulazi se zovu Adversarijalni napadi. Otkriće ovakvih napada je dovelo do razvoja različitih metoda koje formalno dokazuju da neuronska mreža zadovoljava odredjeno svojstvo (npr. robusnost na adversarijalne napade). Neke od tih metoda su zasnovane na SMT rešavačima [2], linearnim aproksimacijama i apstraktnoj interpretaciji [3], [4], [5]. Rad "Abstract Domain for certifying Neural Networks" je primer primene apstraktne interpretacije.

## Rešenje

## Eksperimentalni rezultati

## Zaključak

## Reference

[1]: Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining and Harnessing Adversarial Examples  
[2]: Guy Katz, Clark W. Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. 2017. Reluplex: An Efficient SMT Solver
for Verifying Deep Neural Networks.  
[3]: Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin Vechev. 2018. AI2:
Safety and Robustness Certification of Neural Networks with Abstract Interpretation
[4]: Matthew Mirman, Timon Gehr, and Martin Vechev. 2018. Differentiable Abstract Interpretation for Provably Robust Neural
Networks  
[5]: Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 2018a. Fast and Effective Robustness
Certification
