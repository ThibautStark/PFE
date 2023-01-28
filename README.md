# Neural Network Watermarking

## A bit of context 
In early 2023, the norm of developping a CNN is to copy and paste the weights of already trained CNN. There is, at this date, no way to recognize the training owner yet the dataset and the resources invested in its training represent its value. <br />
Like visual watermarking, neural network watermarking aims to embed the mark of ownership of the resources of a network in the network itself. <br />

This work was carried out by STARK Thibaut & BRUN Loïs for an end of engineering school project (PFE) in Telecom SudParis in the major High Tech Imaging under the supervision of Carl DE SOUSA TRIAS and Mihai MITREA and based on those two papers : <br />
[Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://dl.acm.org/doi/10.1145/3196494.3196550) <br />
[Digital Watermarking for Deep Neural Networks](https://arxiv.org/abs/1802.02601) <br />

## Black and whitebox watermarking

For now, only the two following methods exist for neural network watermarking. <br />

![HowtoWatermark](readme_images/6.PNG "HowtoWatermark") <br />

## Attacks on the watermark
They are two main attacks on our watermark that we are trying to resist from. Both are commonly use after copy and pasting the weights of an already trained neural network. It is then important to study their effects on our watermark. <br />

Thoses two attacks are the following : <br />
![AttacksOnTheWatermark](readme_images/15.PNG "AttacksOnTheWatermark")  <br />
For more information on finetuning please see : <br />
[What is Fine-tuning in Neural Networks?](https://www.baeldung.com/cs/fine-tuning-nn) <br />
For more information on pruning please see : <br />
[Pruning Neural Networks](https://towardsdatascience.com/pruning-neural-networks-1bb3ab5791f9)  <br />

## The watermark triangle
We carrying out a watermarking, one has always to keep in mind those 3 principle to keep as high as possible <br />

![Watermark triangle](readme_images/5.PNG "Watermark triangle")

## Our pratical application

For our study, we chose to use a simple form of resnet on the cifar10 dataset. We made this choice for its simplicity in usage and how much this dataset and this network architecture are widespread in the deeplearning community.  <br />
![Practical Application](readme_images/12.PNG "Practical application")  <br />

With those choices and commun hyperparameters, we obtain after a simple training without watermarking thoses results, which will serve as our control trial <br />
![ResultsWithoutWatermark](readme_images/16.PNG "ResultsWithoutWatermark")

# Blackbox watermarking

## The theory

![BlackBoxTheory1](readme_images/7.PNG "BlackBoxTheory1")

![BlackBoxTheory2](readme_images/8.PNG "BlackBoxTheory2")

## Our hyperparameters

![BlackBoxApplication](readme_images/13.PNG "BlackBoxApplication")

## Our results

![BlackBoxResultsWithoutAttacks](readme_images/17.PNG "BlackBoxResultsWithoutAttacks")

![BlackBoxResultsPruning](readme_images/18.PNG "BlackBoxResultsPruning")

![BlackBoxResultsFinetuning](readme_images/19.PNG "BlackBoxResultsFinetuning")


# Whitebox watermarking

## The theory

![WhiteBoxTheory1](readme_images/9.PNG "WhiteBoxTheory1")

![WhiteBoxTheory2](readme_images/10.PNG "WhiteBoxTheory2")

![WhiteBoxTheory3](readme_images/11.PNG "WhiteBoxTheory3")

So if we recap, the idea behind whitebox watermarking is the following :

![WhiteBoxRecap](readme_images/WhiteBoxRecap.PNG "WhiteBoxRecap")


## Our hyperparameters

![WhiteBoxApplication](readme_images/14.PNG "WhiteBoxApplication")

## Our results

![WhiteBoxResultsWithoutAttacks](readme_images/20.PNG "WhiteBoxResultsWithoutAttacks")

![WhiteBoxResultsWithoutAttacks2](readme_images/21.PNG "WhiteBoxResultsWithoutAttacks2")

![WhiteBoxResultsFinetuning](readme_images/22.PNG "WhiteBoxResultsFinetuning")

![WhiteBoxResultsPruning](readme_images/23.PNG "WhiteBoxResultsPruning")

# Conclusion

![Conclusion](readme_images/24.PNG "Conclusion")

## Sources & thanks
![Sources & thanks](readme_images/26.PNG  "Sources & thanks")
