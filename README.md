# Ingenuity-Trajectory
Theses codes are all the steps involved in the calculation of the trajectory of Ingenuity using the pictures it took during its flights

<p align=center>
  <img src="https://github.com/AstroAure/Ingenuity-Trajectory/blob/main/Flight9-HiRISE-GIF.gif" alt="GIF created by this code of Ingenuity Flight 9 trajectory"/>
</p>

## Description
1. [Ingenuity spotted](/Code/1-Ingenuity_spotted.py) : Finds Ingenuity, given a template, on every picture during a flight
2. [Target generation](/Code/2-Target_generation.py) : Generates intersteing targets on the ground
3. [Feature matching](/Code/3-Feature_matching.py) : Matches these generated targets between two pictures to find relative terrain motion
4. [Movement acquisition](/Code/4-Movement_acquisition.py) : Calculates Ingenuity's movement relative to the ground between two pictures
5. [Trajectory drawing](/Code/5-Trajectory_drawing.py) : Draws the full trajectory of the flight
6. [Trajectory on HiRISE](/Code/6-Trajectory_on_HiRISE.py) : Draws the full trajectory on a given background map made by HiRISE
7. [HiRISE+NavCam](/Code/7-HiRISE+NavCam.py) : Creates a GIF of the flight with NavCam pictures and live orbital follow on the HiRISE map
