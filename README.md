# Ingenuity-Trajectory
Theses codes are all the steps involved in the calculation of the trajectory of Ingenuity using the pictures it took during its flights
![GIF created by this code of Ingenuity Flight 9 trajectory][(Flight9-HiRISE-GIF.gif)

## Description
- 1-Ingenuity spotted : Finds Ingenuity, given a template, on every picture during a flight
- 2-Target generation : Generates intersteing targets on the ground
- 3-Feature matching : Matches these generated targets between two pictures to find relative terrain motion
- 4-Movement acquisition : Calculates Ingenuity's movement relative to the ground between two pictures
- 5-Trajectory drawing : Draws the full trajectory of the flight
- 6-Trajectory on HiRISE : Draws the full trajectory on a given background map made by HiRISE
- 7-HiRISE+NavCam : Creates a GIF of the flight with NavCam pictures and live orbital follow on the HiRISE map
