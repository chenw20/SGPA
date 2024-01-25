IMDB dataset: download [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Additional dependencies to run IMDB experiments:
- pandas - 1.4.3
- transformers - 4.18.0

ECE/MCE reported in the paper are computed according to this [script](https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing#scrollTo=w1SAqFR7wPvs). Note according to this script, ECE/MCE are computed based on the differences between predicted probabilities and the labels for all classes (not just the max-prob class).
