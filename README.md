# ReDST
The code for our ACL paper "Rethinking Dialogue State Tracking with Reasoning"

## How to run?
ReDST is the two-stage approach. First it generate the turn belief using a bert-based sequence labeling model, the code of which is in the folder of turn_belief_gen. Second, it perform reasoning over the generated turn_belief in the first stage, the code of which is in the folder ReDST. Note that, the second stage is independent of the first stage. You can use any sequence labelling model to generate the labels and input to the  second stage.

1. Go to the folder turn_belief_gen, run the python script by the order of its initial number. 
2. Go to the folder ReDST, first you should check the configure file config.yaml to change the hyper-parameters, then run the script by the order of its initial number.

### Acknowledgement
This project is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/lizi-git/ReDST/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
