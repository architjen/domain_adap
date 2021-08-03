# Domain adaptation  
Domain adaptation is the subfield of Machine Learning and Transfer Learning, it involves in creating a mapping function between the training source and target source when both the distribution differs. eg: Say you train your model on the images taken from DSLR camera and you test them on the images taken from webcam.  

SA = Subspace alignment, please refer Fernando et al to understand the algorithm. 
EROT = Entropic regularised transport implementation


— There are 2 ‘.py’ files with Subspace alignment and Entropic regularized optimal transport implementation  


— SA.py file is the comparison of [SA + 1-NN] algorithm with simple 1-NN  
Score 1 is of 1-NN  
Score 2 is of SA + 1-NN  

— EROT.py file is the comparison of SA algo and EROT  
Score 1 is of SA  
Score 2 is of EROT   

- dslr.mat & webcam.mat are the dataset to test the algorithms  
