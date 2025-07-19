from dataclasses import dataclass

@dataclass
class PruneConfig:
    is_prune: bool
    pruned_layer: int
    image_token_start_index: int
    image_token_length: int
    max_num_trunction: int
    reduction_ratio: float
    pivot_image_token: int
    pivot_text_token: int

    _instance = None

    @classmethod
    def instance(cls, is_prune: bool = True, **kwargs):
        if cls._instance is None:
            if is_prune:
                cls._instance = cls(
                    is_prune=is_prune,
                    pruned_layer=kwargs.get("pruned_layer", 2),
                    image_token_start_index=kwargs.get("image_token_start_index", 0),
                    image_token_length=kwargs.get("image_token_length", 0),
                    max_num_trunction=kwargs.get("max_num_trunction", 0),
                    reduction_ratio=kwargs.get("reduction_ratio", 0.778),
                    pivot_image_token=kwargs.get("pivot_image_token", 4),
                    pivot_text_token=kwargs.get("pivot_text_token", 4),
                )
            else:
                cls._instance = cls(
                    is_prune=is_prune,
                    pruned_layer=-1,
                    image_token_start_index=-1,
                    image_token_length=-1,
                    max_num_trunction=-1,
                    reduction_ratio=-1,
                    pivot_image_token=-1,
                    pivot_text_token=-1,
                )
        return cls._instance
