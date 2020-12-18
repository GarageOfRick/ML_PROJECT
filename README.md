# ML_PROJECT
To run the repair badnet1 using find pruning:
python3 repair_badnet1_fine_prune.py <clean validation data directory> <poisoned data> <test data> <model directory>.

E.g., python3 repair_badnet1_fine_prune.py clean_validation_data.h5 sunglasses_poisoned_data.h5 clean_test_data.h5 CSAW-HackML-2020-master/models/sunglasses_bd_net.h5

To run the repair badnet2 using find pruning:
python3 repair_badnet2_fine_prune.py <clean validation data directory> <test data> <model directory>.

E.g., python3 repair_badnet2_fine_prune.py clean_validation_data.h5 clean_test_data.h5 CSAW-HackML-2020-master/models/anonymous_bd_net.h5
