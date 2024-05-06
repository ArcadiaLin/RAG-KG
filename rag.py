from rank_bm25 import BM25Okapi
from functools import lru_cache
import json
import queue
from fuseki_kg import *
from chat_bots import *

# prompts
initialize_prompt = """你是一个金融领域助手，你接下来的回答将严格按照我所提供的资料和要求进行作答，并且无需包含任何多余的解释，是否理解，如果理解请返回1"""
question_prompt = """问题：{}\n这个问题被记作问题1，你现在无须回答这个问题1，请识别这个问题1语句中的命名实体，请以列表的格式作答，即["实体1", "实体2"]"""
triplets_ask_prompt = """现在向你提供以下三元组信息，其基本格式为[[三元组1], [三元组2]],具体的三元组为：{}\n你需要判断以上信息是否足以回答问题1，如果可以，请返回2，并回答问题1；如果不可以，则返回3"""
triplets_complement_prompt = """现在我向你补充一下三元组信息，基本格式不变，三元组为：{}\n基于目前提供的所有信息，是否能够回答问题1,如果可以，请返回2，并回答问题1；如果不可以，则返回3"""
topT_prompt = """基于以上所有三元组，尝试回答问题1，如果无法准确回答，请总结上述三元组信息"""

def levenshtein_distance(s1, s2):
    @lru_cache(None)
    def min_distance(i, j):
        if i == 0 or j == 0:
            return max(i, j)
        elif s1[i - 1] == s2[j - 1]:
            return min_distance(i - 1, j - 1)
        else:
            return 1 + min(min_distance(i, j - 1),    # Insert
                           min_distance(i - 1, j),    # Remove
                           min_distance(i - 1, j - 1) # Replace
                           )
    return min_distance(len(s1), len(s2))


class rag():
    def __init__(self, fuseki: Fuseki, chat_bot, question:str, prune_mode:str, debug_mode=None, topT=3, topN=6):
        self.fuseki = fuseki
        self.chat_bot = chat_bot
        self.question = question
        self.reason_log = []
        self.chat_log = []
        self.topN = topN
        self.topT = topT
        self.triples = queue.Queue()
        self.expandable_nodes = queue.Queue()
        self.name_queue = queue.Queue()
        self.prune_mode = prune_mode
        self.debug_mode = debug_mode

    def chat(self, prompt: str) -> str:
        response: str = self.chat_bot.chat(prompt)
        self.chat_log.append({prompt: response})
        return response
    
    def name_recognition(self):
        recognition_response: str = self.chat(question_prompt.format(self.question))
        name_list = json.loads(recognition_response.replace("\'", "\""))
        for name in name_list:
            self.name_queue.put(name)

    def match_text_with_triplets_bm25(self, text: str, triplets: list) -> list:
        corpus = [triplet[1] for triplet in triplets]
        tokenized_corpus = [[char for char in doc] for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query = [char for char in text]
        scores = bm25.get_scores(query)
        sorted_triples = [triplets[i] for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)]
        return sorted_triples[:self.topN]

    def match_text_with_triplets_Levenshtein(self, text: str, triplets: list) -> list:
        combined_strings = [(' '.join(triple), triple) for triple in triplets]
        distances = [(levenshtein_distance(combined, text), original) for combined, original in combined_strings]
        sorted_triples = [original for _, original in sorted(distances, key=lambda x: x[0])]
        return sorted_triples[:self.topN]

    def triple_filter(self, triple: list) -> list:
        if self.prune_mode == "bm25":
            triple = self.match_text_with_triplets_bm25(self.question, triplets=triple)
        elif self.prune_mode == "lev":
            triple = self.match_text_with_triplets_Levenshtein(self.question, triplets=triple)
        return triple

    def entity_search_by_name(self, name):
        query_triple = self.fuseki.query_by_name(name)
        if query_triple is None:
            return
        triples = self.triple_filter(query_triple['triple'])
        self.triples.put(triples)
        for item in query_triple['related_node']:
            self.expandable_nodes.put(item)

    def expand_nodes(self):
        new_list = []
        while not self.expandable_nodes.empty():
            expandabel_id = self.expandable_nodes.get()
            query_triple = self.fuseki.query_by_id(expandabel_id)
            if query_triple is None:
                continue
            triples = self.triple_filter(query_triple['triple'])
            self.triples.put(triples)
            new_list += query_triple['related_node']
        for item in list(set(new_list)):
            self.expandable_nodes.put(item)

    def initialize(self):
        chat_bot_empty_log = self.chat_bot.messages
        initialize_response = self.chat_bot.chat(initialize_prompt)
        if initialize_response[0] == '1':
            return True
        else:
            self.chat_bot.messages  = chat_bot_empty_log
            self.initialize()

    def reason(self):
        self.initialize()
        self.name_recognition()
        while not self.name_queue.empty():
            item = self.name_queue.get()
            self.entity_search_by_name(item)
        triple = self.triples.get()
        response = self.chat(triplets_ask_prompt.format(json.dumps(triple, ensure_ascii=False)))
        count = 1
        if response[0] == '2':
            return response[1:]
        else:
            self.expand_nodes()
            while not self.triples.empty() and count < self.topT:
                triplet = self.triples.get()
                response = self.chat(triplets_complement_prompt.format(json.dumps(triplet, ensure_ascii=False)))
                if response[0] == "2":
                    return response[1:]
                count += 1
            return  self.chat(topT_prompt)


fuseki_config = {
    'endpoint_url': "fuseki_endpoint",
    'username': 'fuseki_user',
    "password": 'fuseki_passwd'
}
templateLLM_dir = "Qwen|Llama3|Yi dir"
glm_dir = "chatGLM dir"
question = "question"

if __name__ == "__main__":
    fuseki = Fuseki(**fuseki_config)
    # chat_bot = GLMChat(model_file)
    chat_bot = TemplateChatBot(templateLLM_dir)
    rag_test = rag(fuseki=fuseki, chat_bot=chat_bot, question=question, prune_mode="bm25", debug=debug)
    answer = rag_test.reason()
    print(type(answer))