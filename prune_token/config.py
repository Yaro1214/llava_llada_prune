from dataclasses import dataclass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PruneConfig(metaclass=Singleton):
    def __init__(self):
        self.is_prune: bool = False
        self.pruned_layer: int = -1
        self.reduction_ratio: float = -1.0
        self.pivot_image_token: int = -1
        self.pivot_text_token: int = -1

        self.image_token_start_index: int = -1
        self.image_token_length: int = -1
        
        self.suffix_length: int = -1
        self.text_length: int = -1
        self.gen_length: int = -1

        self.current_block: int = 0
        self.current_step: int = 0

    @classmethod
    def new_instance(cls, is_prune=False, **kwargs):
        ins = cls()
        if is_prune:
            ins.is_prune = True
            ins.pruned_layer = kwargs.get("pruned_layer", 2)
            ins.reduction_ratio = kwargs.get("reduction_ratio", 0.778)
            ins.pivot_image_token = kwargs.get("pivot_image_token", 4)
            ins.pivot_text_token = kwargs.get("pivot_text_token", 4)
            
            ins.image_token_starimt_index= kwargs.get("image_token_start_index", 0)
            ins.image_token_length= kwargs.get("image_token_length", 0)

        else:
            ins.is_prune = False
            ins.pruned_layer = -1
            ins.reduction_ratio = -1.0
            ins.pivot_image_token = -1
            ins.pivot_text_token = -1
            ins.init()
        return ins

    def init(self):
        self.current_block = -1
        self.current_step = -1

    # def set(self,suffix_length=None, text_length=None, gen_length=None,image_token_starimt_index=None):
    #     if suffix_length is not None:
    #         self.suffix_length = suffix_length
    #     if text_length is not None:
    #         self.text_length = text_length
    #     if gen_length is not None:
    #         self.gen_length = gen_length
    #     if image_token_starimt_index is not None:
    #         self.image_token_start_index=image_token_starimt_index

    def set(self, **kwargs):
        for attr in ["suffix_length", "text_length", "gen_length", "image_token_start_index"]:
            if attr in kwargs and kwargs[attr] is not None:
                setattr(self, attr, kwargs[attr])

    def update_block(self):
        self.current_block += 1

    def update_step(self):
        self.current_step += 1

    def get_current_block(self):
        return self.current_block

    def get_current_step(self):
        return self.current_step
