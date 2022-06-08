# Supervisory Safety System: ROS2 Version

## Usage

+ Clone the repository
+ Inside the Data folder, make folders for Vehicles, Kernels, and Dynamics (they are ignored by the git ignore)
+ Create kernel by running the KernelGenerator script in the Generator folder
+ Start the fqtenth_gym_ros simulator
+ Run the SafetyTrainer script to train an agent using the safety system for 10 laps (enough for the agent to converge).
+ Test the agent by running the RosEnv script. (note that a random planner and a pure pursuit planner can also be tested here by switching the planner used in the __init__ function. The use of the supervisor can be turned on or off by setting the supervision flag.)

## Notes 
+ Inside the `config/` folder, there are two kinds of config files: 
    1. `config/config_file.yaml`: these are the fixed vehicle parameters, the tuneable parameters and the kernel generation parameters. These are generally fixed and do not change. 
    2. `config/testing_params.yaml`: these are the ros parameters for the parameter server that the node uses that control how the node operates.

