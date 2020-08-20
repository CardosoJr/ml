# NNs Building recipe 

Adapted from Karpathy's blog (https://karpathy.github.io/2019/04/25/recipe/)


### EDA 
* Get to know the data 
* Check for inconsistencies / biases / data leakage 

### E2E Model testing / validation
* Easy and fast way to test and compare models to benchmarks / baselines

#### Tips 
* **Fix random seed**
* **Start from simple models / simplify code**
* **Check your data just before calling the model**: usefull to catch preprocessing / data augmentation bugs
* **Significant digits to eval metrics**
* **Check initial loss:** if final layer is correct, you should see ´´´ -log(1 / n_classes) ´´´ on a softmax at initialization 
* **Initialize layers well:** if regression and dataset mean is 50, initialize final layer's bias to 50. If dataset is imbalanced 1:10, initialize final layer's bias to predict 1:10 probability on logits
* **Monitor metrics that are human interpretable** for sanity check (besides standard ones). Use domain knowledge to build those metrics 
* **Input-independent baseline:** mess up your data and check if them model is worse than when using the correct data (e.g. hardcoded everything to zero and train the model)
* **Overfit your model:** use only one batch with small size (e.g. 2) and check if your model is able to overfit. You should achieve zero train / test loss
* **Check bias-variance tradeoff**: increase the size of your model a bit, check if training loss is decreasing as it should. 
* **Visualize Prediction Dynamics**: plot predictions for a specific test batch during training. Useful to getting insight on how your model is behaving (could use libs like shap to help visualize) 
* **Check backprop gradients**: sometimes you can mess up the tensor (batch_size, ...) and mix the data across the batch dimension. Could change you loss to sum up the output of **i-th** sample, so you'll get non-zero gradients only for the **i-th** sample
* **Start specific, generalize later**: it's easier to make your model / code to a specific case first. After everything is working, generalize to all cases.


### Overfit 
* You now have a pipeline working withou bugs on preprocessing and simple models
* Create a model that can overfit on your data. Then start regularizing it to improve validation loss

#### Tips
* **Get a working / good NN architecture**: get the most related paper or benchmark and replicate the architecture. Don't try at first to create a new, never-seen-before, architecture. Custom may come later to beat this first model
* **Adam**: start with adam which works fine out-of-the-box. SGD could beat adam performance, but most times only with good fine tunning on its hyperparameters. 
* **Build up complexity iteratively**: add one model improvement at a time, so you can track your performance changes. Start with constant LR, than tune and improve to LR schedule
* **Leave it Training**: try training your model for huge time, even when it seens like is overfitting (double descent loss curves)

### Regularize
* Now you have a powerfull model, it's time to regularize it to improve validation loss

#### Tips
* **Get more data**
* **Data augment**: half-fake data, clever data augmentation (resulted from simulations), GANs
* **Pretrain**: if possible, use pretrained networks. Unsupervised pretraining have not shown good results in Computer Vision, but in NLP it has.
* **Smaller input dimensionality**: remove unusefull features or feature with high noise 
* **Reduce Model Size**: using domain knowledge, it's possible to reduce model size in some cases.
* **Decrease the batch size**
* **Dropout**
* **Weight Decay**
* **Early Stopping**: sometimes big models (that overfit easily) with good early stopping mechanism can lead to better results

### Tune

#### Tips
* **Tunning**: don't use grid search, it's better to use random search or more fancy optimizers with TPE or bayesian optimization. 
* **Ensembles**
