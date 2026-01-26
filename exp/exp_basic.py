from models import M3LLM_Llama3, M3LLM_Llama, M3LLM_Gpt2, M3LLM_Opt_1b, M3LLM_Qwen


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'M3LLM_Llama': M3LLM_Llama,
            'M3LLM_Llama3': M3LLM_Llama3,
            'M3LLM_Gpt2': M3LLM_Gpt2,
            'M3LLM_Opt_1b': M3LLM_Opt_1b,
            'M3LLM_Qwen': M3LLM_Qwen
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
