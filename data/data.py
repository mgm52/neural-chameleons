from collections import defaultdict
from typing import Dict, List


### SAMPLES ###
class PromptResp:
    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response

class PromptRespRating:
    def __init__(self, prompt: str, response: str, ratings: Dict[str, float]):
        self.prompt = prompt
        self.response = response
        self.ratings = ratings

### DATASETS ###
class PosNegData:
    def __init__(self, pos_dataset: List[PromptResp], neg_dataset: List[PromptResp]):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        self.__balance_to_same_len()
    
    def __balance_to_same_len(self):
        min_len = min(len(self.pos_dataset), len(self.neg_dataset))
        self.pos_dataset = self.pos_dataset[:min_len]
        self.neg_dataset = self.neg_dataset[:min_len]

class PosNegDataByCategory:
    def __init__(self, categories: Dict[str, PosNegData]):
        self.categories = categories
    
    @classmethod
    def from_prompt_resp_dict(cls, prompt_resp_dict: Dict[str, List[PromptResp]]):
        # Pos dataset = existing list of promptresp
        # Neg dataset = all other promptresps
        categories = defaultdict(lambda: PosNegData(pos_dataset=[], neg_dataset=[]))
        for category, prompt_resps in prompt_resp_dict.items():
            categories[category].pos_dataset = prompt_resps
            for other_category, other_prompt_resps in prompt_resp_dict.items():
                if other_category != category:
                    categories[category].neg_dataset.extend(other_prompt_resps)
            categories[category].__balance_to_same_len() # TODO: shuffle first
        return cls(categories)

    @classmethod
    def from_ratings(cls, ratings_list: List[PromptRespRating], max_neg_rating: float = 0.25, min_pos_rating: float = 0.75):
        categories = defaultdict(lambda: PosNegData(pos_dataset=[], neg_dataset=[]))
        
        for rating_obj in ratings_list:
            prompt_response = PromptResp(rating_obj.prompt, rating_obj.response)
            for category, rating_value in rating_obj.ratings.items():
                if rating_value <= max_neg_rating:
                    categories[category].neg_dataset.append(prompt_response)
                elif rating_value >= min_pos_rating:
                    categories[category].pos_dataset.append(prompt_response)
        
        for category in categories:
            categories[category].__balance_to_same_len()
        
        return cls(categories)