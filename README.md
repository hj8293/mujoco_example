# MuJoCo Slipping Artifact Demo

This script demonstrates a simple grasping scenario where **two spherical ‘fingertips’** squeeze and hold a **box-shaped object**.  
With the default configuration (including `impratio` and `noslip_iterations`), the object stays stably grasped.  
If you remove these two parameters, the same setup will begin **slipping**, even though the friction should theoretically be sufficient.

## How to Run
python mujoco_example.py

This launches a MuJoCo viewer that prints contact forces and slip status.

## How to Reproduce the Artifact
1. In the generated XML (Line 41), the default stable version contains:
   <option ... impratio="100000" noslip_iterations="10"/>

2. Remove `impratio` and `noslip_iterations`, then run again:
   python mujoco_example.py

You should now observe visible slipping, confirming that stability depends on MuJoCo-specific parameters, not physical properties.

## Note
This example isolates how MuJoCo’s default contact settings can cause slipping; other simulators may behave differently.
