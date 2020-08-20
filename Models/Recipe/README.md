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
* **Significant digits to eval metrics**
* **Check initial loss:** if final layer is correct, you should see ´´´ -log(1 / n_classes) ´´´ on a softmax at initialization 
* **Initialize layers well:** if regression and dataset mean is 50, initialize final layer's bias to 50. If dataset is imbalanced 1:10, initialize final layer's bias to predict 1:10 probability on logits
* **Monitor metrics that are human interpretable** for sanity check (besides standard ones). Use domain knowledge to build those metrics 
* **Input-independent baseline:** mess up your data and check if them model is worse than when using the correct data (e.g. hardcoded everything to zero and train the model)
* **Overfit your model:** use only one batch with small size (e.g. 2) and check if your model is able to overfit. You should achieve zero train / test loss
& 

