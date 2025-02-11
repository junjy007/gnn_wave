from dataclasses import dataclass, field

@dataclass
class Config:
    ## TASK DEF OVERALL
    forward_time: int = 100 # go forward for 100 time steps
    snapshots: int = 1
    output_nodes: str = "all" # for now, all, we can work on this later.
  
    ## DATA
    # path
    data_dir: str = "data"
    data_fname: str = "wave.nc"
    normalization: str = "none"
        
    # train/test
    train_period: tuple = (0, 1000) # max_t is non-inclusive
    test_period: tuple = (1000, 1400)

    ## GRAPH MODEL
    #kernal size 
    convolution_kernels: list = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024])
    # Define the number of nearest neighbors you want to connect
    neighbourhood_size:int = 5
    node_var_observ: list = field(default_factory=lambda: 
        ['hs', 'ub_bot', 'wlen', 'pwave_bot', 'tpeak', 'dirm'])
    node_var_target: list = field(default_factory=lambda:
        ['hs', 'ub_bot', 'wlen', 'pwave_bot', 'tpeak', 'dirm'])
    
    ## TRAINING
    learning_rate:float = 0.001
    max_epoches:int = 10

    data_shuffle_seed:int = 42
    train_shuffle:bool = True
    # data loader
    train_batch_size: int=4
    test_batch_size: int=16
