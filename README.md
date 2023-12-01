# Learn to Predict Human Manipulation of Large-Sized Objects Through Interactive Motions

Heelo, welcome to our project repository!

## Dataset Access

To access our dataset, please visit this [Google Drive link](https://drive.google.com/drive/folders/174k-o7UFuIZg8BsZRbDGcAbVskKsF36r?usp=sharing).

## Visualizing Dataset Samples

To view samples from our dataset, use the following command:


    python visData.py


## Obtaining SMPL Parameters

To extract SMPL parameters of human skeleton motion:

1. Download the dataset and place it in the `./data/` directory.
2. Follow the configuration settings in `render_mesh.py`.
3. Run the following command to utilize SMPLify ([SMPLify website](https://smplify.is.tue.mpg.de)) for extracting SMPL parameters. This process may be time-consuming.

    
    python render_mesh.py
    

4. The SMPL meshes and parameters will be saved in `./data/SAMPLE_NAME/SMPL_result/`.
5. To view the SMPL joint motions, use:

    
    python visData.py --smpl_joint True
    

## Simulation Tool

Included in this repository is a basic simulator for testing the physical properties of various object layouts. Run it using:


    python Simulator/Simulate_Demo.py

## Future Updates

Stay tuned for more materials and updates to this project.
