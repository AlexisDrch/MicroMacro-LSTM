### LSTM Mic Mac
Micro Macro (LSTM - MicMac)

A new chained-model that considers sequence inputs with different granularities to model their relationship before classifying.

Assumption: Data can have long-term dependencies at different granularity. In a book, understanding a sequence of character might help classifying the sentences and sequence of sentences might help classifying the whole chapter. But what about using those different granularities and learn their relashionship to get a better overall classification of the book/chapter or video topic.

In that regard, MicMac is a model that takes as input both a macro and a micro sequence of feature vectors, using different window-length and step in time (for time-series) different sequence of frames (for video classification) or different input granularities (for NLP). Those two (or n) different sequences are extracted at different scales and used to train two (or n) separate LSTMs. 

Outputs are merged and the overall relationship between micro/macro is then learned through two fully connected layers (to learn the micro-macro relationship) and a last Softmax layer is used for classification.

<i> Future work: macro's input are micro's output to get this chain, compact single pass of backprop between different macro and micro model. </i>
