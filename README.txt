There are considerable amount of Python files within this submission. Only a subsection handle processes required for the experiments of this project, others are scripts to organise these processes.
The processes of this project are structured in a modular, sequential pipeline. To run this pipeline, execute one of the following files:
- config1.py
- config2.py
- config3.py
- config4.py
- configTracks.py
Each of these files contain a different configuration of the pipeline, described within the file. The pipeline sections in general terms are as follows:
1. Preprocessing - parameters can be tuned to determine how data is preprocessed.
	This includes preprocessing.py
2. Movement estimation - 1 of 3 optical flow configurations are run to summarise the movement visible in each data item.
	This includes FrameOpticalFlow.py and tracksOpticalFlow.py
3. Data set evaluation - features are extracted from the optical flow data and saved in datasets.
	This includes evaluateDataSetsV1.py, evaluateDataSetsV2.py, evaluateDataSetsV3.py, and tracksEVD.py
4. Estimation - a classifier is produced and its performance is assessed.
	This includes estimation.py
5. Feature Performance evaluation - further statistics are obtained from the logs produced in estimation.
	This inclues featurePerformanceStats.py and featurePercPerformanceStats.py

Each of these files require command line arguments to operate. The format of these can be found in any of the previously mentioned configuration scripts.

In order to run correctly, the pipeline must be executed in order. This should be done through a congifuration script.
Additionally, each stage requires a dataset to operate. If none is found, the program cannot run. This data is expected to be found in specific, named folders within the project directory.
The folder containing the experiment code is expected to be located within the project directory, at the same level as other folders containing specific data required for the operation of certain programs.
Each stage of the pipeline produces data required for the next stage.

Several external libraries are required including the following:
- moviepy
- sklearn
- numba
- numpy
- cv2



