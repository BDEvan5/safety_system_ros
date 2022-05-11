# Supervisory Safety System: ROS2 Version

## Usage

+ Clone the repository
+ Inside the Data folder, make folders for Vehicles, Kernels, and Dynamics (they are ignored by the git ignore)
+ Create kernel by running the KernelGenerator script in the Generator folder
+ Start the fqtenth_gym_ros simulator
+ Run the SafetyTrainer script to train an agent using the safety system for 10 laps (enough for the agent to converge).
+ Test the agent by running the RosEnv script. (note that a random planner and a pure pursuit planner can also be tested here by switching the planner used in the __init__ function. The use of the supervisor can be turned on or off by setting the supervision flag.)



