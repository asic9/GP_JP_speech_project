This folder contains the speech recognition project files:
> Presentation.
> Model training notebook (second.ipynb).
> Main real time processing script (main.py).
> Simplistic audio clip testing script that either selects a training 
file or records a 1 second clip (speech_guesser.ipynb).
> Label encoder used to convert the one-hot encoded output from the model to words.
> Two models (.h5 files) with different structures shown in the presentation. 
The file names give an indication of the model structure (it was trained only on X_train!). The more 
complex one was used in the demonstration. "nfft512" means that nfft=512 was used when computing the MFCCs.
> Helvetica.ttf is just the font used to print the words in the real-time program.


# A suggestion for a more interesting project.
This project was quite interesting, but the main issue is that 99% of the effort was not 
actually related to neural networks directly. Rather, the main part was setting up the data, 
preprocessing, speech testing, dealing with memory overflow issues... 
For the neural network, basically the simplest CNN structure immediately worked with good results 
and after somewhat random minor changes it could successfully be used for prediction. The second 
(more complicated 3 block + LSTM) structure was basically a flex-overkill just to see what happens 
(turns out - not much difference, just less epochs needed to train). 
This is somewhat underwhelming for a course about neural networks. 

So I thought I would come up with a suggestion for a more interesting project. 
Instead of starting from scratch, other students can start with what we did already and 
attempt letter-based prediction. I was toying with this idea at the start, but the effort 
required to do this from scratch is too big for the course. 
If starting from our code, the following things would need to be done:
> Find a dataset with lots of labeled audio (e.g. https://commonvoice.mozilla.org/en/datasets).
> Preprocess and save the data in batches (if training data is too big to fit in one big array). 
> Find a way to accomodate input MFCCs of different time lengths.
> Write a model that also uses Connectionist Temporal Classification (CTC) loss. Ask ChatGPT about it.
> Feed the data in smaller chucks to the model (solves the GPU memory bottleneck). For this the function 
train_ds = tf.data.Dataset.from_generator 
from second.ipynb can be adapted and a descently large batch size could be used for speed.
> The word-formation would need to be figured out independently (we didn't get there yet).