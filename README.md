# Neural Network Watermarking

## A bit of context 
In early 2023, the norm of developping a CNN is to copy and paste the weights of already trained CNN. There is, at this date, no way to recognize the training owner yet the dataset and the resources invested in its training represent its value. <br />
Like visual watermarking, neural network watermarking aims to embed the mark of ownership of the resources of a network in the network itself. <br />

This work was carried out by STARK Thibaut & BRUN Loïs for an end of engineering school project (PFE) in Telecom SudParis in the major High Tech Imaging under the supervision of [Carl DE SOUSA TRIAS](https://github.com/Carldst) and Mihai MITREA and based on those two papers : <br />
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

The main idea behind blackbox watermarking is to embed an unatural error inside the neural network training so that, we the error is detected we can say for sure that this is our watermark <br />

In order to do that, we transform part of a class of our dataset in a unatural way, for instance by writing "TEST" on it, then we put it in the most visually opposed class. After that data manipulation, we can simply test if a similarly modified image put into the network is returned to as the predicted error <br />
![BlackBoxTheory1](readme_images/7.PNG "BlackBoxTheory1")

For our application, we chose to put "HTI" onto images of planes and to move then to the horses label. Planes have a different form and background texture and color than horses. <br />
![BlackBoxTheory2](readme_images/8.PNG "BlackBoxTheory2")

## Our hyperparameters

For our experiments, we chose theses hyperparameters since they are communly used for this kind of training <br />

![BlackBoxApplication](readme_images/13.PNG "BlackBoxApplication")

## Our results

![BlackBoxResultsWithoutAttacks](readme_images/17.PNG "BlackBoxResultsWithoutAttacks")

![BlackBoxResultsPruning](readme_images/18.PNG "BlackBoxResultsPruning")

![BlackBoxResultsFinetuning](readme_images/19.PNG "BlackBoxResultsFinetuning")


# Whitebox watermarking

## The theory

The idea behind whitebox watermarking is to embed the watermark into the weights themselves. In order to carry out such a task, we need to have access to the weights of a chosen layer. <br />

In order to change the weights to embed our watermark, we are changing the loss itself. In order to do that we add to the original loss our watermark loss with a regularization term. <br />
![WhiteBoxTheory1](readme_images/9.PNG "WhiteBoxTheory1") <br />

This watermark loss is created by using a X_key. This X_key is created with the size of our original watermark (which is a (Tx1) tensor made out of 1 and 0)) and has to be secret. Only with this key you can detect the watermark. X_key is created with randomly generated numbers between 0 and 1.   <br />
![WhiteBoxTheory2](readme_images/10.PNG "WhiteBoxTheory2")<br />

With that X_key, we can project the layer's weigths onto it then do a binary cross entropy in order to get our previously mentionned watermark loss <br />
![WhiteBoxTheory3](readme_images/11.PNG "WhiteBoxTheory3") <br />

So if we recap, the idea behind whitebox watermarking is the following : <br />

![WhiteBoxRecap](readme_images/WhiteBoxRecap.PNG "WhiteBoxRecap") <br />


## Our hyperparameters

For our experiments, we chose theses hyperparameters since they are communly used for this kind of training. Also we randomly chose to watermark the 4th convolution layer but the watermark can be applied anywhere else. <br />

![WhiteBoxApplication](readme_images/14.PNG "WhiteBoxApplication") <br />

## Our results

After embedding our watermarks onto the weights of the 4th conv layer during the training. We then reproject thoses weigths onto our initial X_key, when then compare the results of this projection to our threshold which is here 0.5. Then compare this tensor of size of (Tx1) made out of 1 and 0 to our original watermark. <br />

We can see here that there is a 100% retriaval accuracy of our watermark tensor from thoses watermarked weights. <br />

Also, we notice that even the watermarking of the loss heavily affect the performance of the model in the first epochs, after several epochs the difference between the original loss and the watermarked loss is insignificant <br />

![WhiteBoxResultsWithoutAttacks](readme_images/20.PNG "WhiteBoxResultsWithoutAttacks")

The demonstrate the importance of the X_key, we try to deciefer our watermarked weights with a X_key that is different from our own. Here, we take another randomized X_key (which was not used for watermarking the weights). <br />
The difference between 100% and 80% tells us that only the original X_key can achieve optimal retrivial performances. As such, the secrecy and ownership of the original X_key is paramount to claim ownership of the network. <br />
![WhiteBoxResultsWithoutAttacks2](readme_images/21.PNG "WhiteBoxResultsWithoutAttacks2") <br />

We finetune with 10% of the training epochs to simulate a real attack of the watermark. <br />
We notice that there is still a 100% retrieval accuracy and as such, whitebox being robust to finetuning <br />
![WhiteBoxResultsFinetuning](readme_images/22.PNG "WhiteBoxResultsFinetuning")

We chose to prune the first layer of the network but could have chosen any other. We notice that the retrieval accuracy is affected by this attacks and as such we can say that whitebox watermarking is not robust to pruning <br />
More importantly, if by a stroke of unluck, the attacker attack the watermarked layer, the watermark completely disappear. <br />
![WhiteBoxResultsPruning](readme_images/23.PNG "WhiteBoxResultsPruning")

# Conclusion

We can then, after some experimentation draw this chart for the watermarking of neural network. <br />
![Conclusion](readme_images/24.PNG "Conclusion")

Also, an important point to note is that the watermark naturally decays of the increase of epochs of the attacks. As such, even though one originaly put a certain quantity for the training of a network and watermark it, someone else can put the same amount of effort to retrain this network without the first one being able to claim ownership.
As such, legal and moral implicatoin of the ownership of network training is not resolved by this study. <br />
## Sources & thanks
This project was carried out on the basis of those two articles. 
We would like to profoundly thank our supervisor and tutor for the precious help they offered in all the aspects of this project <br />
![Sources & thanks](readme_images/26.PNG  "Sources & thanks")
