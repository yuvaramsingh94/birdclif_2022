# birdclif_2022


This is a two stage model.

** stage 1 ** 
- SED model for detecting bird and no bird call.
- used the opensource bird, no bird model dataset.
- this dataset has some issue with label. they have lot a label noise. better clean it.

** Stage 2 **
- SED model with 152 bird.
- percentile based thresholding.
- tfimm model save it locally and load it without internet in kaggle.