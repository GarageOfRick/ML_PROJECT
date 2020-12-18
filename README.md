# ML_PROJECT
To run the repair badnet1 using find pruning:
python3 repair_badnet1_fine_prune.py clean_validation_data poisoned_data test_data bad_net.

E.g., python3 repair_badnet1_fine_prune.py clean_validation_data.h5 sunglasses_poisoned_data.h5 test_data.h5 CSAW-HackML-2020-master/models/sunglasses_bd_net.h5

To run the repair badnet2 using find pruning:
python3 repair_badnet2_fine_prune.py clean_validation_data test_data bad_net.

E.g., python3 repair_badnet2_fine_prune.py clean_validation_data.h5 test_data.h5 CSAW-HackML-2020-master/models/anonymous_bd_net.h5

Backdoor data neuron activation:
![](images/backdoorede%20data%20neuron%20activation.png)

Clean data neuron activation:
![](images/clean%20data%20neuron%20activation.png)
