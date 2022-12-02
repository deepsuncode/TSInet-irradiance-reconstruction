This ReadMe explains the requirements and getting started to run the Total Solar Irradiance reconstruction using the Deep Learning network TSInet.

Prerequisites:

Python, Tensorflow, and Cuda:
The initial work and implementation of TSInet was done using Python version 3.6.8, Tensorflow 1.14.0 and GPU Cuda version 10.0.
Therefore, in order to run the default out-of-the-box models to run some reconstruction, you should use the exact version of Python and Tensorflow. 
There is also another model created for Python v 3.8.6 , Cuda version 10.2 and Tensorflow 2.4. You may also use this model with the exact versions of Python and Tensorflow.
Other versions are not tested, but they should work if you have the environment set properly to run deep learning jobs.

Python Packages:
A) For Python 3.6.8 and Tensorflow 1.14
The following python packages and modules are required to run TSInet:
numpy
scikit-learn
keras
keras_self_attention
pandas
scipy
tensorflow==1.14
tensorflow-gpu==1.14


B) For Python >= 3.8.6 and Tensorflow 2.4
numpy
keras
keras_self_attention
pandas
sklearn
scipy
tensorflow==2.4
tensorflow-gpu==2.4


To install the required packages, you may use Python package manager “pip” as follow:
1.	Copy the above packages into a text file,  ie “requirements.txt”
2.	Execute the command:
pip install -r requirements.txt
Note: There is a requirements file already created for you to use that includes tensorflow==1.14 which should be used with Python 3.6.8 pip package manager. 
       And another requirement files "requirements_tf_2.4.txt" for Python 3.8.6 and tensorflow 2.4. 
       The files are located in the root directory of the TSInet package.
Note: Python packages and libraries are sensitive to versions. Please make sure you are using the correct packages and libraries versions as specified above.
Also, note that Python >= 3.8.x does not support Tensorflow version 1.14, therefore, please follow the versions specified above.
Note: The Tensorflow backend warning is thrown when you use tensorflow 2 and its safe to ignore the warning.

Cuda Installation Package:
You may download and install Cuda v 10.0 from https://developer.nvidia.com/cuda-10.0-download-archive
And 10.1 from https://developer.nvidia.com/cuda-10.1-download-archive-base


Package Structure
After downloading the zip files from https://web.njit.edu/~wangj/TSInet/, unzip the files into a directory so that the TSInet package includes the following folders and files:
 
 ReadMe.txt              	 - this ReadMe file.
 requirements.txt        	 - includes Python required packages for Python version 3.6.8 and tensorflow version 1.14.
 requirements_tf_2.4.txt 	 - includes Python required packages for Python version >= 3.8.6 and tensorflow version 2.4.
 models                  	 - directory for newly trained models.
 default_model         		 - includes default trained model used during the initial work of TSInet.
 logs                   	 - includes the logging inforation.
 test_data             	 	 - includes a list of TSI data sets that can be used for reconstruction.
 train_data             	 - includes the SORCE TSI dataset that is used for building the TSInet network model.
 results                	 - will include the reconstruction result file(s)
 tsinet_reconstructor.py         - Python program to test/reconstruct a trained model.
 tsinet_train.py         	 - Python program to train a model.
 tsinet_utils.py        	 - utilities program used by the test and training programs.
 
Running a Test/Reconstruction Task:
1.	To run a test/reconstruction, you should use the existing data sets from the "test_data” directory. 
 tsinet_test.py is used to run the test/reconstruction. 
Type: python tsinet_test.py -h will show you the available options as follows:
	usage: tsinet_reconstructor.py [-h] -n DATASET_NAME [-r RESULT_FILE_NAME] [-e EPOCHS] [-z NUMBER_OF_DAYS] [-k K_STEPS] [-l VERBOSE]                                                                
                                                                                                                                                                                                   
optional arguments:                                                                                                                                                                                
  -h, --help            show this help message and exit                                                                                                                                            
  -n DATASET_NAME, --dataset_name DATASET_NAME                                                                                                                                                     
                        The data set name for the data you want to reconstruct. The required dataset name can be one of the datasets: TCTE, SATIRE-S, NRLTSI2, SATIRE-M                            
  -r RESULT_FILE_NAME, --result_file_name RESULT_FILE_NAME                                                                                                                                         
                        Full path to the result file to save the reconstructed data. Default is: tsinet_result_<dataset name>.csv. File will be saved in the results directory.                    
  -e EPOCHS, --epochs EPOCHS                                                                                                                                                                       
                        The number of epochs to use when re-fitting the data into the model. Default is 10                                                                                         
  -z NUMBER_OF_DAYS, --number_of_days NUMBER_OF_DAYS                                                                                                                                               
                        The number of days to reconstruct for the given reconstruction data set. Default is only 30 days of the reconstruction data set. 
                        Note: depending on the size of the data this may take long time, therefore, it's recommended to provide smaller number such as 
                        10-30 days at least for testing and before you run full data set, to use the  full size of the data, provide 0 days..                                                                                                                                    
  -k K_STEPS, --k_steps K_STEPS                                                                                                                                                                    
                        Number of k-steps to refit the model.                                                                                                                                      
  -l VERBOSE, --verbose VERBOSE                                                                                                                                                                    
                        Verbose level to print processing logs, default is False.  
                                                                                                                                        
You may change the options as you wish to test/reconstruct the desired test data.

2. Examples to run test/reconstruction job:
 python tsinet_reconstructor.py -n TCTE # to run a reconstruction for the dataset TCTE with default options.
 
  python tsinet_reconstructor.py -n TCTE -r my_result_file.csv -z 10  
  To run a reconstruction job for the dataset TCTE, result file name is my_result_file.csv for 10 days of reconstruction.
Note: when you construct SATIRE-M, please make sure to use multiple of 10 years in days such as 365*10*<1,2,3...> to because SATIRE-M is decadal.
      In addition, using the entire dataset will be extremely slow and requires a powerful GPU machine.
Running a Training Task:
1.	tsinet_train.py is used to run the training. 
Type: python tsinet_train.py -h will show you the available options as follows:
	usage: tsinet_train.py [-h] [-e EPOCHS] [-a ATTENTION_LAYER] [-u NUM_UNITS] [-n NORMALIZE_DATA] [-l VERBOSE] [-v MODEL_VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train the network.
  -a ATTENTION_LAYER, --attention_layer ATTENTION_LAYER
                        Add the additional layer to generate more focused input and output. Default is True
  -u NUM_UNITS, --num_units NUM_UNITS
                        The number of LSTM units to use during learning. Default is 400
  -n NORMALIZE_DATA, --normalize_data NORMALIZE_DATA
                        Normalize the TSI data before processing, default is True.
  -l VERBOSE, --verbose VERBOSE
                        Verbose level to print processing logs, default is False.
  -v MODEL_VERBOSE, --model_verbose MODEL_VERBOSE
                        Verbose level to print processing logs, default is 2, one line per epoch. 1 is a progress bar for each epoch, and 0 is silent.


2.	Examples to run a training:
	python tsinet_train.py	#to run a training job with default parameters.

	python tsinet_train.py -e 30 -v 1 
	To run a training job with number of epochs set to 30 and verbose set to 1 to show progress bar for each epoch.

