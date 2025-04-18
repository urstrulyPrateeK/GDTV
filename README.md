# Gender Detection through Voice
- **PreProcessing**
  - Mozilla Common Voice data downloaded from [Kaggle](https://www.kaggle.com/mozillaorg/common-voice)
  - Extracted files with Gender attribute
  - Removed extra data from one gender to keep the dataset balanced
  - Processed the data and extracted **Mel Spectogram Feature** using Librosa
  - Saved all the features in **.npy** files and all the information in **finaldata.csv**
- **Training**
  - Loaded all the data
  - Created a model with 5 fully connected layers with 20% dropout rate
  - Fit the model with Early Stopping with patience 5
  - Saved the model 
- **Prediction**
  - Using mic module from [StackOverflow](https://stackoverflow.com/a/6743593/15324584)
  - Load the weights to model, predict and return the results
- **GUI**
  - Created a random grid of colours using HTML and CSS on Flask
  - Added button for turning on the mic
  - Animation using jQuery
