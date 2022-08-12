# ISLES22 ATLAS: Sample Docker
This repository serves as a template for your to produce a Docker container with your model.
Your model should be trained and loadable at this stage.  
There are three important files: for you to modify:
- `requirements.txt` - Python dependencies for your model.  
  Python packages specified in `requirements.txt` will be installed in the container's Python environment when it is built.
- `process.py` - Modify the section to load and call your model.  
  Load your model and use it to make predictions on the input.
- `Dockerfile` - Add the files needed to run your model (model weights, code, etc.)

Once complete, you can run `build.sh` to build the container, and `export.sh` to package it for upload.
The original source code for the algorithm container was generated with evalutils version 0.3.1.
