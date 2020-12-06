# from abc import ABCMeta

class BaseTrainer:
    """
    trainer will run in following procedure:
    train():
        for e in epoch:
            begin_epoch()
            train_loop():
                train_step()
            after_train_loop()
            validate()
            end_epoch() ## detectron combine this step to after_train
    """
    def __init__(self):
        """
        init logging, model, data_loader, optimizer, scheduler, ... 
        """
        pass
    
    def train(self):
        """
        run training procedure
        """
        pass
    
    def begin_epoch(self):
        """
        procedure at the begining of each epoch
        such as reset logger for each epoch
        """
        pass
    
    def train_loop(self):
        """
        training loop in each epoch
        """
        pass
    
    def train_step(self):
        """
        usualy involve: 
        get next batch -> forward -> backward -> optimizer, scheduler step -> update log 
        """
        pass
    
    def after_train_loop(self):
        """
        usualy involve evaluating training epoch, logging
        """
        pass
    
    def validate(self):
        """
        run validation in val_dataset
        """
        pass
    
    def end_epoch(self):
        """
        close epoch logging files
        save checkpoint
        """
        pass
    

class BaseEvaluator:
    """
    Evaluation procedure
    run_eval():
        before_loop()
        for batch in dataloader:
            step()
        after_loop()
    """
    def __init__(self):
        """
        get monitor, data loader, model, 
        """
        pass
    
    def before_loop(self):
        """
        reset monitor
        """
        pass
    
    def step(self):
        """
        infer step
        """
        pass
    
    def after_loop(self):
        """
        run computer performance matrics
        """
        pass
    
    def run_eval(self):
        """
        run
        """
        pass