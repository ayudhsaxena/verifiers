from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class TextArenaRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", "response"]),
                 env_parser: XMLParser = XMLParser(fields=["output"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]

   
    

