# ML_PROJECT

## Fine-pruning
To run the repair badnet1 using fine-pruning:

`python3 repair_badnet1_fine_prune.py clean_validation_data poisoned_data test_data bad_net`

E.g., `python3 repair_badnet1_fine_prune.py clean_validation_data.h5 sunglasses_poisoned_data.h5 test_data.h5 CSAW-HackML-2020-master/models/sunglasses_bd_net.h5`

To run the repair badnet2 using fine-pruning:

`python3 repair_badnet2_fine_prune.py clean_validation_data test_data bad_net`

E.g., `python3 repair_badnet2_fine_prune.py clean_validation_data.h5 test_data.h5 CSAW-HackML-2020-master/models/anonymous_bd_net.h5`

Results show:

Backdoor data neuron activation:

<img src="images/backdoorede%20data%20neuron%20activation.png" width="300" height="400">

Clean data neuron activation:

<img src="images/clean%20data%20neuron%20activation.png" width="300" height="400">

Pruned_neurons to prediction accuracy on validation data with bad net 1:

<img src="images/bd_net1_acc.png" width="250" height="200">

Pruned_neurons to prediction accuracy on validation data with bad net 2:

<img src="images/bd_net2_acc.png" width="250" height="200">

## STRIP

### Usage

To repair any BadNet using STRIP:

```shell
python repair_badnet_strip.py clean_validation_data poisoned_data clean_test_data badnet badnet_weights
```

E.g.

```shell
python repair_badnet_strip.py data/clean_validation_data.h5 data/sunglasses_poisoned_data.h5 data/clean_test_data.h5 models/sunglasses_bd_net.h5 models/sunglasses_bd_weights.h5
```

### Result Quick View

#### Human Face Images

Clean sample

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/Sample_Clean.png)

Sunglasses sample

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/Sample_Sunglasses.png)

Eyebrows sample

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/Sample_Eyebrows.png)

Perturbation demonstration

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/Sample_Perturbation.png)

#### BadNet1

Entropy distribution of perturbed clean (benign) and poisoned (trojan) samples.

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/BadNet1, n_test=2000 ,n_perturb=100.png)

False Rejection Rate (FRR), False Acceptance Rate (FAR) relationship curve.

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/FAR_FRR_BadNet1.png)

Accuracy performance on clean validation dataset *clean_validation_data.h5*: 91.63419069888282%

#### BadNet2

Entropy distribution of perturbed clean samples.

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/BadNet2, n_test=2000, n_perturb=100,200.png)

Accuracy performance on clean validation dataset *clean_validation_data.h5*: 76.66926474408938%

#### BadNet3

Entropy distribution of perturbed clean (benign) and poisoned (trojan) samples.

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/BadNet3, n_test=2000 ,n_perturb=100.png)

False Rejection Rate (FRR), False Acceptance Rate (FAR) relationship curve.

![](/Volumes/MacintoshSD/3_DEV/ML_PROJECT/strip/images/FAR_FRR_BadNet3.png)

Accuracy performance on clean validation dataset *clean_validation_data.h5*: 80.54906036199879%

For more detailed explanation and performance about code, please refer to Jupyter Notebook ***9163 Project STRIP.ipynb*** (PDF version also available) and our project report.