# Feed Forward Neural Network
This project contains a C++ implementation of FFNN.

File structure:
- `src` - contains source code
    - `activation_functions` - implementation of various activation functions
    - `csv` - csv reader and writer
    - `data_manager` - train/val split, random shuffle, batch generator
    - `data_structures` - matrix
    - `network` - network configuration, network itself (forward/backward pass, ...)
    - `optimizers` - adam, sgd
    - `schedulers` - learning rate scheduler
    - `statistics` - accuracy, cross entropy (loss), argmax, stats (weight stats) printers
    - `utils` - hyper-parameter configuration testing utility functions
    
If you are using Windows with WSL, change `-Ofast` to `O3`. Do so even if you encounter strange behaviour (nan, inf, etc...).
